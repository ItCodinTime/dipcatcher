"""Reported backtest cost components omit turnover the cash account pays."""

from datetime import UTC, datetime

import polars as pl
import pytest

from quant_fund.backtest.engine import run_backtest
from quant_fund.config.loader import load_config


def test_backtest_cost_breakdown_omits_turnover_bps(tmp_path) -> None:
    """Flat book: NAV drop equals commission + spread + impact + turnover.

    ``metrics`` sums only the first three. With ``bps_per_turnover=25`` the
    gap equals the turnover cash. Default configs set that rate to 0, so
    sealed runs are unchanged. The backtest metric keys are left alone
    because adding a key would change analytics-export digests.
    """
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.costs.frictionless = False
    cfg.costs.commission_bps = 1.0
    cfg.costs.half_spread_bps = 0.0
    cfg.costs.impact_y = 0.0
    cfg.costs.bps_per_turnover = 25.0
    cfg.costs.borrow_bps_per_year = 0.0
    cfg.costs.participation_limit = 1.0
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 1.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    when = [
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 2, tzinfo=UTC),
        datetime(2024, 1, 3, tzinfo=UTC),
    ]
    bars = pl.DataFrame(
        {
            "security_id": ["A", "A", "A"],
            "event_time": when,
            "open": [100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0],
            "close_total_return": [100.0, 100.0, 100.0],
            "volume": [1_000_000.0, 1_000_000.0, 1_000_000.0],
            "adv": [1e12, 1e12, 1e12],
            "vol_20": [0.0, 0.0, 0.0],
            "source": ["synthetic", "synthetic", "synthetic"],
        }
    )
    weights = pl.DataFrame(
        {
            "event_time": [when[0]],
            "security_id": ["A"],
            "target_weight": [0.5],
        }
    )
    initial = 100_000.0
    result = run_backtest(bars, weights, cfg, initial_nav=initial)
    reported = (
        float(result.metrics["commission"])
        + float(result.metrics["spread"])
        + float(result.metrics["impact"])
    )
    nav_drop = initial - float(result.equity["nav"][-1])
    notional = sum(
        abs(float(qty)) * float(px)
        for qty, px in zip(result.fills["quantity"], result.fills["price"], strict=True)
    )
    turnover = notional * 25.0 / 1e4
    assert turnover > 0.0
    assert nav_drop == pytest.approx(reported + turnover)
    assert nav_drop == pytest.approx(turnover + notional * 1.0 / 1e4)
    assert reported == pytest.approx(nav_drop - turnover)
