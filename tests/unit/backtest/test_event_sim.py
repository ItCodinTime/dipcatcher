"""Event-driven execution simulator: parity, fills, costs, constraints."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from quant_fund.backtest.engine import run_backtest
from quant_fund.backtest.event_sim.clock import EventClock, EventKind
from quant_fund.backtest.event_sim.constraints import ConstraintBook
from quant_fund.backtest.event_sim.costs import (
    almgren_chriss_impact,
    alpaca_equity_schedule,
    quote_execution_cost,
    schedule_fee,
    spread_crossing_cost,
)
from quant_fund.backtest.event_sim.fills import advance_queue, allocate_vwap, walk_book
from quant_fund.backtest.event_sim.sensitivity import (
    execution_sensitivity,
    format_sensitivity_table,
)
from quant_fund.backtest.event_sim.simulator import EventSimSpec, run_event_backtest
from quant_fund.backtest.fast_replay import run_backtest_fast
from quant_fund.config.loader import load_config
from quant_fund.config.models import CostConfig
from quant_fund.execution.almgren_chriss import expected_shortfall_ac
from quant_fund.schemas.order_book import BookLevel, OrderBookSnapshot


def _cfg(tmp_path, *, frictionless: bool = True):
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.costs.frictionless = frictionless
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 1.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    cfg.costs.participation_limit = 1.0
    return cfg


def _bars(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def _day(sid: str, day: int, price: float, *, source: str = "file", hour: int = 0) -> dict:
    when = datetime(2024, 1, day, hour, tzinfo=UTC)
    return {
        "security_id": sid,
        "event_time": when,
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "close_total_return": price,
        "volume": 1_000_000.0,
        "adv": 100_000_000.0,
        "vol_20": 0.02,
        "source": source,
    }


def _assert_close(sim_result, ref) -> None:
    sim = sim_result.result
    assert sim.equity.height == ref.equity.height
    if sim.equity.height:
        np.testing.assert_allclose(
            sim.equity["nav"].to_numpy(),
            ref.equity["nav"].to_numpy(),
            atol=1e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            sim.equity["turnover"].to_numpy(),
            ref.equity["turnover"].to_numpy(),
            atol=1e-12,
            rtol=0.0,
        )
    assert sim.fills.height == ref.fills.height
    if sim.fills.height:
        left = sim.fills.sort(["fill_time", "security_id"])
        right = ref.fills.sort(["fill_time", "security_id"])
        np.testing.assert_allclose(
            left["quantity"].to_numpy().astype(float),
            right["quantity"].to_numpy().astype(float),
            atol=1e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            left["price"].to_numpy().astype(float),
            right["price"].to_numpy().astype(float),
            atol=1e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            left["fee"].to_numpy().astype(float),
            right["fee"].to_numpy().astype(float),
            atol=1e-12,
            rtol=0.0,
        )
    for key in ("total_return", "sharpe", "commission", "spread", "impact"):
        if key not in ref.metrics:
            continue
        got = sim.metrics.get(key)
        exp = ref.metrics.get(key)
        if isinstance(exp, (int, float)) and isinstance(got, (int, float)):
            if np.isnan(exp):
                assert np.isnan(got)
            else:
                assert got == pytest.approx(exp, abs=1e-12)
    assert sim.metrics["research_only"] is True
    assert sim.metrics["live_pnl_claim"] is False
    assert sim_result.min_cash >= -1e-8


def test_clock_orders_market_before_fill_before_signal() -> None:
    clock = EventClock(seed=7)
    clock.schedule(1, EventKind.SIGNAL, {})
    clock.schedule(0, EventKind.FILL, {})
    clock.schedule(0, EventKind.MARKET, {})
    clock.schedule(0, EventKind.ORDER, {})
    seen = [(event.bar_index, event.kind) for event in clock.run(lambda _event: None)]
    assert seen == [
        (0, EventKind.MARKET),
        (0, EventKind.FILL),
        (0, EventKind.ORDER),
        (1, EventKind.SIGNAL),
    ]


def test_zero_latency_zero_cost_matches_vectorized_backtest(tmp_path) -> None:
    cfg = _cfg(tmp_path, frictionless=True)
    rows = []
    for day, price in ((1, 100.0), (2, 100.0), (3, 110.0), (4, 105.0)):
        rows.append(_day("A", day, price))
        rows.append(_day("B", day, price / 2.0))
    bars = _bars(rows)
    weights = pl.DataFrame(
        {
            "event_time": [
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
            ],
            "security_id": ["A", "B", "A", "B"],
            "target_weight": [0.5, -0.2, 0.1, 0.0],
        }
    )
    ref = run_backtest(bars, weights, cfg, initial_nav=100_000.0)
    fast = run_backtest_fast(bars, weights, cfg, initial_nav=100_000.0)
    sim = run_event_backtest(bars, weights, cfg, EventSimSpec(seed=11), initial_nav=100_000.0)
    _assert_close(sim, ref)
    _assert_close(sim, fast)
    kinds = [event["kind"] for event in sim.events]
    assert "MARKET" in kinds and "SIGNAL" in kinds and "ORDER" in kinds and "FILL" in kinds
    again = run_event_backtest(bars, weights, cfg, EventSimSpec(seed=99), initial_nav=100_000.0)
    np.testing.assert_allclose(
        sim.result.equity["nav"].to_numpy(),
        again.result.equity["nav"].to_numpy(),
        atol=0.0,
        rtol=0.0,
    )


def test_costed_next_open_matches_reference(tmp_path) -> None:
    cfg = _cfg(tmp_path, frictionless=False)
    cfg.costs.impact_y = 0.1
    cfg.costs.commission_bps = 1.0
    cfg.costs.half_spread_bps = 5.0
    cfg.costs.borrow_bps_per_year = 50.0
    rows = [_day("A", day, 100.0 + day) for day in range(1, 6)]
    rows += [_day("B", day, 50.0 + day) for day in range(1, 6)]
    bars = _bars(rows)
    weights = pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC)],
            "security_id": ["A", "B"],
            "target_weight": [0.4, -0.1],
        }
    )
    ref = run_backtest(bars, weights, cfg, initial_nav=250_000.0)
    fast = run_backtest_fast(bars, weights, cfg, initial_nav=250_000.0)
    sim = run_event_backtest(bars, weights, cfg, EventSimSpec(), initial_nav=250_000.0)
    _assert_close(sim, ref)
    _assert_close(sim, fast)


def test_latency_delays_fill_bar(tmp_path) -> None:
    cfg = _cfg(tmp_path, frictionless=True)
    bars = _bars([_day("A", day, 100.0) for day in range(1, 5)])
    weights = pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, 1, tzinfo=UTC)],
            "security_id": ["A"],
            "target_weight": [1.0],
        }
    )
    base = run_event_backtest(bars, weights, cfg, EventSimSpec(), initial_nav=10_000.0)
    delayed = run_event_backtest(
        bars,
        weights,
        cfg,
        EventSimSpec(signal_to_order_bars=1, order_to_exchange_bars=1),
        initial_nav=10_000.0,
    )
    assert base.result.fills["fill_time"][0] == datetime(2024, 1, 2, tzinfo=UTC)
    assert delayed.result.fills["fill_time"][0] == datetime(2024, 1, 4, tzinfo=UTC)
    timed = run_event_backtest(
        bars,
        weights,
        cfg,
        EventSimSpec(signal_to_order=timedelta(days=1), order_to_exchange=timedelta(0)),
        initial_nav=10_000.0,
    )
    assert timed.result.fills["fill_time"][0] == datetime(2024, 1, 3, tzinfo=UTC)


def test_vwap_does_not_exceed_bar_volume(tmp_path) -> None:
    cfg = _cfg(tmp_path, frictionless=True)
    rows = []
    for day in range(1, 4):
        row = _day("A", day, 100.0)
        row["volume"] = 10.0
        row["high"] = 101.0
        row["low"] = 99.0
        rows.append(row)
    bars = _bars(rows)
    weights = pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, 1, tzinfo=UTC)],
            "security_id": ["A"],
            "target_weight": [1.0],
        }
    )
    sim = run_event_backtest(
        bars,
        weights,
        cfg,
        EventSimSpec(fill_model="vwap", vwap_window_bars=2),
        initial_nav=100_000.0,
    )
    assert sim.result.fills.height >= 1
    for qty in sim.result.fills["quantity"].to_list():
        assert abs(float(qty)) <= 10.0 + 1e-9
    # Each exec bar can print at most that bar's volume; the carried target
    # may work several bars, so the sum is bounded by volume times fills.
    assert float(sim.result.fills["quantity"].abs().sum()) <= 10.0 * sim.result.fills.height + 1e-9


def test_l2_walk_does_not_exceed_displayed_size() -> None:
    taken = walk_book("buy", 100.0, [(101.0, 3.0), (102.0, 5.0)])
    assert taken.filled == pytest.approx(8.0)
    assert taken.vwap == pytest.approx((3 * 101.0 + 5 * 102.0) / 8.0)
    assert taken.filled <= taken.displayed + 1e-12
    sold = walk_book("sell", 3.0, [(99.0, 4.0), (98.0, 6.0)])
    assert sold.filled == pytest.approx(3.0)
    assert sold.filled <= sold.displayed + 1e-12


def test_l2_queue_respects_displayed_and_cancel_replace(tmp_path) -> None:
    cfg = _cfg(tmp_path, frictionless=True)
    rows = [_day("A", day, 100.0) for day in range(1, 5)]
    bars = _bars(rows)
    weights = pl.DataFrame(
        {
            "event_time": [
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
            ],
            "security_id": ["A", "A"],
            "target_weight": [1.0, 0.0],
        }
    )
    books: dict[tuple[str, datetime], OrderBookSnapshot] = {}
    for day in range(1, 5):
        when = datetime(2024, 1, day, tzinfo=UTC)
        ask = 101.0 if day < 3 else 103.0
        books[("A", when)] = OrderBookSnapshot(
            security_id="A",
            symbol="A",
            event_time=when,
            available_time=when,
            source="test",
            bids=[BookLevel(price=99.0, size=50.0)],
            asks=[BookLevel(price=ask, size=5.0)],
            depth=1,
        )
    sim = run_event_backtest(
        bars,
        weights,
        cfg,
        EventSimSpec(fill_model="l2_queue", l2_rest_bars=3),
        initial_nav=100_000.0,
        books=books,
    )
    assert sim.result.fills.height >= 1
    for qty in sim.result.fills["quantity"].to_list():
        assert abs(float(qty)) <= 5.0 + 1e-9
    assert sim.cancel_replace_count >= 1
    assert sim.result.metrics["live_pnl_claim"] is False


def test_alpaca_schedule_is_zero_commission_plus_sell_fees() -> None:
    schedule = alpaca_equity_schedule()
    assert schedule_fee(10_000.0, 100.0, is_sell=False, schedule=schedule) == 0.0
    sell = schedule_fee(10_000.0, 100.0, is_sell=True, schedule=schedule)
    assert sell > 0.0
    # $10_000 * $27.80 per $1mm = $0.278 → $0.28; 100 shares * $0.000166 → $0.02 TAF.
    assert sell == pytest.approx(0.30)
    quoted = quote_execution_cost(
        -100.0,
        100.0,
        1e8,
        0.02,
        CostConfig(frictionless=False, commission_bps=0.0, half_spread_bps=0.0, impact_y=0.0),
        schedule=schedule,
        is_sell=True,
    )
    assert float(quoted["commission"]) == pytest.approx(sell)
    assert float(quoted["total"]) >= float(quoted["commission"])


def test_almgren_chriss_matches_expected_shortfall_and_is_calibratable() -> None:
    quantity = 12.5
    eta, gamma, tau = 1.5e-4, 2.0e-5, 0.5
    got = almgren_chriss_impact(quantity, eta=eta, gamma=gamma, tau=tau)
    ref = expected_shortfall_ac(
        np.array([quantity, 0.0]),
        np.array([quantity]),
        arrival=1.0,
        eta=eta,
        gamma=gamma,
        sigma=0.2,
        tau=tau,
    )
    assert got["temporary"] + got["permanent"] == pytest.approx(ref["expected_cost"], abs=1e-12)
    bigger = almgren_chriss_impact(quantity * 2, eta=eta, gamma=gamma, tau=tau)
    assert bigger["total"] >= got["total"]
    assert spread_crossing_cost(1_000.0, 5.0) == pytest.approx(0.5)


def test_fractional_shares_and_min_notional(tmp_path) -> None:
    cfg = _cfg(tmp_path, frictionless=True)
    bars = _bars([_day("A", day, 100.0) for day in range(1, 3)])
    weights = pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, 1, tzinfo=UTC)],
            "security_id": ["A"],
            "target_weight": [1.0],
        }
    )
    whole = run_event_backtest(
        bars,
        weights,
        cfg,
        EventSimSpec(fractional_shares=False),
        initial_nav=250.0,
    )
    assert whole.result.fills["quantity"][0] == pytest.approx(2.0)
    dust = run_event_backtest(
        bars,
        weights,
        cfg,
        EventSimSpec(min_notional=1_000_000.0),
        initial_nav=10_000.0,
    )
    assert dust.result.fills.height == 0


def test_no_negative_cash_without_margin(tmp_path) -> None:
    cfg = _cfg(tmp_path, frictionless=False)
    bars = _bars([_day("A", day, 100.0) for day in range(1, 4)])
    weights = pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, 1, tzinfo=UTC)],
            "security_id": ["A"],
            "target_weight": [1.0],
        }
    )
    sim = run_event_backtest(
        bars, weights, cfg, EventSimSpec(allow_margin=False), initial_nav=1_000.0
    )
    assert sim.min_cash >= -1e-8
    assert sim.result.metrics["cash_rejects"] >= 1


def test_settlement_good_faith_and_pdt(tmp_path) -> None:
    cfg = _cfg(tmp_path, frictionless=True)
    rows = []
    for day in range(1, 5):
        rows.append(_day("A", day, 100.0))
        rows.append(_day("B", day, 100.0))
    bars = _bars(rows)
    weights = pl.DataFrame(
        {
            "event_time": [
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
                datetime(2024, 1, 3, tzinfo=UTC),
                datetime(2024, 1, 3, tzinfo=UTC),
            ],
            "security_id": ["A", "A", "B", "A", "B"],
            "target_weight": [1.0, 0.0, 1.0, 0.0, 0.0],
        }
    )
    sim = run_event_backtest(
        bars,
        weights,
        cfg,
        EventSimSpec(settlement_bars=2, gfv_mode="warn", pdt_mode="off"),
        initial_nav=100_000.0,
    )
    assert sim.gfv_count >= 1
    assert any(item["code"] == "GFV" for item in sim.warnings)
    assert sim.min_cash >= -1e-8

    pdt_rows = [
        _day("A", 1, 100.0),
        _day("A", 2, 100.0, hour=14),
        _day("A", 2, 100.0, hour=15),
    ]
    # _day uses day-of-month; hour distinguishes the session bars on Jan 2.
    pdt_bars = _bars(pdt_rows)
    pdt_weights = pl.DataFrame(
        {
            "event_time": [
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, 14, tzinfo=UTC),
            ],
            "security_id": ["A", "A"],
            "target_weight": [1.0, 0.0],
        }
    )
    warned = run_event_backtest(
        pdt_bars,
        pdt_weights,
        cfg,
        EventSimSpec(pdt_mode="warn", pdt_max_day_trades=0),
        initial_nav=10_000.0,
    )
    assert warned.day_trade_count == 1
    blocked = run_event_backtest(
        pdt_bars,
        pdt_weights,
        cfg,
        EventSimSpec(pdt_mode="block", pdt_max_day_trades=0),
        initial_nav=10_000.0,
    )
    assert blocked.result.fills.height == 1
    assert float(blocked.result.fills["quantity"][0]) > 0.0


def test_constraint_book_buy_rejects_overdraft() -> None:
    book = ConstraintBook(settlement_bars=1, allow_margin=False, settled=50.0)
    assert book.apply_buy("A", 1.0, 80.0, 0) is False
    assert book.settled == pytest.approx(50.0)


def test_sensitivity_table_degrades_with_impact(tmp_path) -> None:
    cfg = _cfg(tmp_path, frictionless=False)
    cfg.costs.impact_y = 0.2
    cfg.costs.half_spread_bps = 5.0
    rows = []
    for day in range(1, 6):
        rows.append(_day("A", day, 100.0 + day, source="SYNTHETIC"))
    bars = _bars(rows)
    weights = pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 3, tzinfo=UTC)],
            "security_id": ["A", "A"],
            "target_weight": [0.5, 0.0],
        }
    )
    report = execution_sensitivity(
        bars,
        weights,
        cfg,
        signal_latencies=(0, 1),
        exchange_latencies=(0,),
        impact_multipliers=(1.0, 2.0),
        initial_nav=100_000.0,
        seed=3,
    )
    assert report["claim"] == "execution_diagnostic_only"
    assert report["live_pnl_claim"] is False
    table = format_sensitivity_table(report)
    assert "EXECUTION DIAGNOSTIC ONLY" in table
    assert "net_pnl" in table and "sharpe" in table
    by_key = {
        (row["signal_to_order_bars"], row["impact_multiplier"]): row for row in report["rows"]
    }
    assert by_key[(0, 2.0)]["total_cost"] >= by_key[(0, 1.0)]["total_cost"]
    assert by_key[(0, 2.0)]["net_pnl"] <= by_key[(0, 1.0)]["net_pnl"] + 1e-8


def test_vwap_allocator_and_queue_never_overfill() -> None:
    slices = allocate_vwap(25.0, [(10.0, 100.0), (10.0, 101.0), (10.0, 102.0)])
    assert [qty for qty, _price in slices] == pytest.approx([10.0, 10.0, 5.0])
    step = advance_queue(
        side="buy",
        remaining=20.0,
        limit_price=100.0,
        queue_ahead=15.0,
        touch_price=100.0,
        touch_size=15.0,
        traded_volume=25.0,
    )
    assert step.filled <= 15.0
    assert step.filled <= 25.0
    replaced = advance_queue(
        side="buy",
        remaining=5.0,
        limit_price=100.0,
        queue_ahead=4.0,
        touch_price=99.0,
        touch_size=8.0,
        traded_volume=30.0,
    )
    assert replaced.replaced is True
    assert replaced.filled == 0.0
    assert replaced.queue_ahead == 8.0
