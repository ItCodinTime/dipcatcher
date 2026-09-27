"""Property checks for the execution simulator.

No negative cash without margin, fills stay inside displayed or traded volume,
and costs are monotone in size.
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.backtest.event_sim.costs import (
    FeeSchedule,
    almgren_chriss_impact,
    quote_execution_cost,
    spread_crossing_cost,
)
from quant_fund.backtest.event_sim.fills import advance_queue, walk_book
from quant_fund.backtest.event_sim.simulator import EventSimSpec, run_event_backtest
from quant_fund.config.loader import load_config
from quant_fund.config.models import CostConfig

_SIZE = st.floats(min_value=0.0, max_value=5_000.0, allow_nan=False, allow_infinity=False)
_POS = st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False)


@given(q1=_SIZE, q2=_SIZE, price=_POS)
@settings(max_examples=40, deadline=None)
def test_execution_costs_monotone_in_size(q1: float, q2: float, price: float) -> None:
    small, large = sorted((q1, q2))
    config = CostConfig(
        frictionless=False,
        commission_bps=1.0,
        half_spread_bps=4.0,
        impact_y=0.15,
        bps_per_turnover=0.5,
    )
    schedule = FeeSchedule(name="bps", commission_bps=1.0)
    kwargs = dict(
        price=price,
        adv_dollars=1e7,
        sigma=0.02,
        config=config,
        schedule=schedule,
        is_sell=False,
        eta=1e-6,
        gamma=1e-6,
        tau=1.0,
    )
    low = float(quote_execution_cost(small, **kwargs)["total"])
    high = float(quote_execution_cost(large, **kwargs)["total"])
    assert high + 1e-9 >= low
    assert spread_crossing_cost(large * price, 4.0) + 1e-9 >= spread_crossing_cost(
        small * price, 4.0
    )
    assert (
        almgren_chriss_impact(large, eta=1e-5, gamma=1e-5)["total"] + 1e-9
        >= (almgren_chriss_impact(small, eta=1e-5, gamma=1e-5)["total"])
    )


@given(
    qty=_SIZE,
    sizes=st.lists(_POS, min_size=1, max_size=4),
)
@settings(max_examples=40, deadline=None)
def test_book_walk_never_exceeds_displayed(qty: float, sizes: list[float]) -> None:
    levels = [(100.0 + index, size) for index, size in enumerate(sizes)]
    filled = walk_book("buy", qty, levels)
    assert filled.filled <= sum(sizes) + 1e-9
    assert filled.filled <= qty + 1e-9
    if filled.filled > 0.0:
        assert filled.vwap > 0.0


@given(
    remaining=_POS,
    ahead=_SIZE,
    touch=_POS,
    traded=_SIZE,
)
@settings(max_examples=40, deadline=None)
def test_queue_fill_never_exceeds_displayed_or_volume(
    remaining: float, ahead: float, touch: float, traded: float
) -> None:
    step = advance_queue(
        side="sell",
        remaining=remaining,
        limit_price=50.0,
        queue_ahead=ahead,
        touch_price=50.0,
        touch_size=touch,
        traded_volume=traded,
    )
    assert step.filled <= remaining + 1e-9
    assert step.filled <= touch + 1e-9
    assert step.filled <= traded + 1e-9
    assert step.queue_ahead >= -1e-9


@given(
    w1=st.floats(min_value=0.0, max_value=0.4, allow_nan=False, allow_infinity=False),
    w2=st.floats(min_value=0.0, max_value=0.4, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=12, deadline=None)
def test_simulator_cash_stays_non_negative_without_margin(w1: float, w2: float) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.costs.frictionless = False
    cfg.costs.impact_y = 0.05
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 1.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    cfg.costs.participation_limit = 1.0
    rows = []
    for day in range(1, 5):
        for sid, price in (("A", 80.0 + day), ("B", 40.0 + day)):
            rows.append(
                {
                    "security_id": sid,
                    "event_time": datetime(2024, 1, day, tzinfo=UTC),
                    "open": price,
                    "close": price,
                    "close_total_return": price,
                    "volume": 1_000_000.0,
                    "adv": 50_000_000.0,
                    "vol_20": 0.02,
                    "source": "SYNTHETIC",
                }
            )
    bars = pl.DataFrame(rows)
    weights = pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC)],
            "security_id": ["A", "B"],
            "target_weight": [w1, w2],
        }
    )
    sim = run_event_backtest(
        bars,
        weights,
        cfg,
        EventSimSpec(allow_margin=False, seed=1),
        initial_nav=50_000.0,
    )
    assert sim.min_cash >= -1e-8
    assert sim.result.metrics["live_pnl_claim"] is False
