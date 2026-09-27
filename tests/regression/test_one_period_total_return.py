"""A single marked bar is a return against starting capital, not a zero."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from quant_fund.backtest.engine import run_backtest
from quant_fund.backtest.fast_replay import run_backtest_fast
from quant_fund.config.models import AppConfig


def _config() -> AppConfig:
    cfg = AppConfig()
    cfg.data.root = "/tmp/dipcatcher-one-period-return"
    cfg.costs.frictionless = True
    cfg.costs.borrow_bps_per_year = 0.0
    cfg.costs.participation_limit = 1.0
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 1.0
    cfg.risk_gate.max_order_notional = 1e15
    cfg.risk_gate.max_participation = 1.0
    cfg.risk_gate.max_predicted_vol = 100.0
    return cfg


def test_one_equity_row_total_return_matches_ending_nav() -> None:
    t0 = datetime(2024, 1, 2, tzinfo=UTC)
    t1 = datetime(2024, 1, 3, tzinfo=UTC)
    bars = pl.DataFrame(
        {
            "security_id": ["A", "A"],
            "event_time": [t0, t1],
            "open": [100.0, 110.0],
            "close": [100.0, 110.0],
            "close_total_return": [100.0, 121.0],
            "volume": [1_000_000.0, 1_000_000.0],
            "adv": [1e12, 1e12],
            "vol_20": [0.02, 0.02],
            "source": ["file", "file"],
        }
    ).with_columns(pl.col("event_time").cast(pl.Datetime("us", "UTC")))
    weights = pl.DataFrame(
        {"event_time": [t0], "security_id": ["A"], "target_weight": [1.0]}
    ).with_columns(pl.col("event_time").cast(pl.Datetime("us", "UTC")))
    initial = 1_000_000.0
    cfg = _config()
    for engine in (run_backtest, run_backtest_fast):
        result = engine(bars, weights, cfg, initial_nav=initial)
        assert result.equity.height == 1
        end = float(result.equity["nav"][0])
        assert end == 1_100_000.0
        assert float(result.metrics["total_return"]) == end / initial - 1.0
        assert result.metrics["research_only"] is True
        assert result.metrics["live_pnl_claim"] is False
