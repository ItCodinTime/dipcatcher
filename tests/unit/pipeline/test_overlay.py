"""Portfolio overlay: vol targeting, drawdown governor, causal engine wiring."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_fund.backtest.overlay import (
    CompositeScaler,
    DrawdownGovernor,
    VolTargetScaler,
)
from quant_fund.backtest.perp_engine import run_perp_backtest
from quant_fund.config.loader import load_config

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def test_vol_target_scales_down_high_vol() -> None:
    s = VolTargetScaler(target_ann_vol=0.10, window=10, periods_per_year=8766.0)
    nav = 1e5
    for i in range(12):
        nav *= 1 + (0.02 if i % 2 == 0 else -0.02)  # ±2% per bar → huge vol
        s.observe(T0 + timedelta(hours=i), nav)
    assert s.factor() < 0.2


def test_vol_target_unchanged_until_window_fills() -> None:
    s = VolTargetScaler(target_ann_vol=0.10, window=50)
    for i in range(10):
        s.observe(T0 + timedelta(hours=i), 1e5)
    assert s.factor() == 1.0


def test_drawdown_governor_ramps() -> None:
    g = DrawdownGovernor(dd_soft=0.03, dd_hard=0.05, floor=0.25)
    g.observe(T0, 100.0)
    g.observe(T0, 99.0)  # 1% dd → no cut
    assert g.factor() == 1.0
    g.observe(T0, 96.0)  # 4% dd → mid-ramp: 1 - 0.5*0.75 = 0.625
    assert g.factor() == pytest.approx(0.625)
    g.observe(T0, 94.0)  # 6% dd → floor
    assert g.factor() == 0.25
    g.observe(T0, 120.0)  # new peak → resets
    assert g.factor() == 1.0


def test_composite_multiplies() -> None:
    c = CompositeScaler([DrawdownGovernor(dd_soft=0.0, dd_hard=0.1, floor=0.5)])
    c.observe(T0, 100.0)
    c.observe(T0, 95.0)  # 5% dd → 1 - 0.5*0.5 = 0.75
    out = c.scale(T0, {"A": 1.0})
    assert out["A"] == pytest.approx(0.75)


def test_scaler_is_causal_inside_engine(tmp_path) -> None:
    """A governor reading the engine's own NAV path must not see the future."""
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.costs.frictionless = True
    cfg.risk_gate.max_name = 10.0
    cfg.risk_gate.max_net = 10.0
    cfg.risk_gate.max_gross = 10.0
    cfg.risk_gate.max_order_notional = 1e12
    # crash mid-path: governor must react AFTER the drawdown, not before
    prices = [100.0] * 30 + [70.0] * 5 + [70.0] * 30
    bars = pl.DataFrame(
        {
            "security_id": ["A"] * len(prices),
            "event_time": [T0 + timedelta(hours=i) for i in range(len(prices))],
            "open": prices,
            "high": [p * 1.001 for p in prices],
            "low": [p * 0.999 for p in prices],
            "close": prices,
            "volume": [1e6] * len(prices),
            "source": ["synthetic"] * len(prices),
        }
    )
    weights = pl.DataFrame(
        {
            "event_time": [T0 + timedelta(hours=i) for i in range(60)],
            "security_id": ["A"] * 60,
            "target_weight": [0.5] * 60,
        }
    )
    gov = DrawdownGovernor(dd_soft=0.05, dd_hard=0.10, floor=0.1)
    res = run_perp_backtest(bars, None, weights, cfg, initial_nav=1e5, scaler=gov)
    plain = run_perp_backtest(bars, None, weights, cfg, initial_nav=1e5)
    # after the crash (flat price), the governed book holds ~floor× the exposure
    gross_g = res.equity["gross"].to_list()
    gross_p = plain.equity["gross"].to_list()
    assert gross_g[-1] < gross_p[-1] * 0.3
    assert res.metrics["live_pnl_claim"] is False
