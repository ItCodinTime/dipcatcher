"""Closed-form + wiring tests for OHLC spread cost calibration."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from quant_fund.backtest.engine import _fast_replay_is_complete, run_backtest
from quant_fund.backtest.fast_replay import run_backtest_fast
from quant_fund.config.models import (
    AppConfig,
    CostConfig,
    ExecutionConfig,
    FillConvention,
    KillSwitchConfig,
    RiskGateConfig,
)
from quant_fund.execution.costs import total_cost
from quant_fund.execution.spread_calibration import (
    abdi_ranaldo_relative_spread,
    corwin_schultz_relative_spread,
    floored_half_spread_bps,
    pit_calibrated_half_spread_bps,
    relative_to_half_spread_bps,
    roll_relative_spread,
)
from quant_fund.research.cost_calibration import (
    COST_CALIBRATION_SCHEMA,
    format_cost_calibration_table,
    run_cost_calibration_trials,
    write_cost_calibration_receipt,
    write_cost_calibration_report,
)

runner = CliRunner()


def _cs_pair_closed_form(h0: float, l0: float, h1: float, l1: float) -> float:
    den = 3.0 - 2.0 * math.sqrt(2.0)
    beta = math.log(h1 / l1) ** 2 + math.log(h0 / l0) ** 2
    gamma = math.log(max(h0, h1) / min(l0, l1)) ** 2
    alpha = (math.sqrt(2.0 * beta) - math.sqrt(beta)) / den - math.sqrt(gamma / den)
    exp_a = math.exp(alpha)
    return 2.0 * (exp_a - 1.0) / (1.0 + exp_a)


def test_corwin_schultz_matches_closed_form_single_pair() -> None:
    h0, l0, h1, l1 = 110.0, 90.0, 112.0, 88.0
    expected = max(_cs_pair_closed_form(h0, l0, h1, l1), 0.0)
    got = corwin_schultz_relative_spread(np.array([h0, h1]), np.array([l0, l1]))
    assert got == pytest.approx(expected, rel=1e-12)


def test_abdi_ranaldo_matches_closed_form_single_pair() -> None:
    # close away from HL mid so eta > 0
    h = np.array([10.0, 12.0])
    lo = np.array([8.0, 10.0])
    c = np.array([9.0, 11.5])  # m0=9, m1=11; eta=(11.5-11)*(9-9)=0 → zero
    assert abdi_ranaldo_relative_spread(h, lo, c) == pytest.approx(0.0)
    c2 = np.array([8.5, 11.8])  # m0=9, m1=11; eta=(11.8-11)*(8.5-9)=0.8*(-0.5)<0 → 0
    assert abdi_ranaldo_relative_spread(h, lo, c2) == pytest.approx(0.0)
    c3 = np.array([8.5, 10.2])  # eta=(10.2-11)*(8.5-9)=(-0.8)*(-0.5)=0.4
    expected = 2.0 * math.sqrt(0.4) / 10.2
    assert abdi_ranaldo_relative_spread(h, lo, c3) == pytest.approx(expected, rel=1e-12)


def test_roll_bounce_closed_form() -> None:
    """Alternating ±1 bounce → γ₁ = −1 → abs spread 2 → relative 2/mid."""
    mid = np.array([10.0, 11.0] * 8)
    assert roll_relative_spread(mid) == pytest.approx(2.0 / 10.5, rel=1e-12)


def test_floored_half_spread_never_below_flat() -> None:
    floor = 5.0
    # relative 0.0002 → half bps = 1.0 < floor → floor wins
    assert floored_half_spread_bps(floor, 0.0002) == pytest.approx(5.0)
    # relative 0.002 → half bps = 10.0 > floor
    assert floored_half_spread_bps(floor, 0.002) == pytest.approx(10.0)
    # undefined estimate → floor
    assert floored_half_spread_bps(floor, float("nan")) == pytest.approx(5.0)
    assert relative_to_half_spread_bps(0.002) == pytest.approx(10.0)


def test_total_cost_honors_half_spread_override() -> None:
    cfg = CostConfig(half_spread_bps=5.0, impact_y=0.0, commission_bps=0.0)
    base = total_cost(100.0, 10.0, 1e9, 0.0, cfg)
    hi = total_cost(100.0, 10.0, 1e9, 0.0, cfg, half_spread_bps=20.0)
    assert base["spread"] == pytest.approx(0.5)  # |1000| * 5 / 1e4
    assert hi["spread"] == pytest.approx(2.0)  # |1000| * 20 / 1e4
    assert hi["spread"] > base["spread"]


def _cfg(**cost_kwargs: object) -> AppConfig:
    costs = {
        "commission_bps": 1.0,
        "half_spread_bps": 1.0,
        "impact_y": 0.0,
        "bps_per_turnover": 0.0,
        "borrow_bps_per_year": 0.0,
        "frictionless": False,
        "participation_limit": 1.0,
        "spread_estimator": "flat",
        "spread_calibration_lookback": 20,
    }
    costs.update(cost_kwargs)
    return AppConfig(
        costs=CostConfig(**costs),  # type: ignore[arg-type]
        risk_gate=RiskGateConfig(
            max_order_notional=1e12,
            max_gross=100.0,
            max_net=100.0,
            max_name=1.0,
            max_participation=1.0,
            max_predicted_vol=100.0,
            stale_price_bars=5,
            stale_model_hours=1e9,
        ),
        execution=ExecutionConfig(fill=FillConvention.NEXT_OPEN, allow_close_auction=False),
        kill_switch=KillSwitchConfig(state="ENABLED"),
    )


def _tiny_ohlc_book() -> tuple[pl.DataFrame, pl.DataFrame]:
    start = datetime(2024, 1, 2, tzinfo=UTC)
    rows = []
    wrows = []
    for t in range(12):
        et = start + timedelta(days=t)
        close = 100.0 + 0.1 * t
        # Wide range so CS >> 1 bp floor
        rows.append(
            {
                "event_time": et,
                "security_id": "A",
                "open": close,
                "high": close * 1.05,
                "low": close * 0.95,
                "close": close,
                "close_total_return": close,
                "volume": 1_000_000.0,
                "adv": close * 1_000_000.0,
                "vol_20": 0.02,
                "source": "SYNTHETIC",
            }
        )
        wrows.append(
            {
                "event_time": et,
                "security_id": "A",
                "target_weight": 0.2 if t % 2 == 0 else 0.8,
            }
        )
    return pl.DataFrame(rows), pl.DataFrame(wrows)


def test_fast_replay_refuses_calibrated_spread_estimator() -> None:
    bars, weights = _tiny_ohlc_book()
    cfg = _cfg(spread_estimator="corwin_schultz")
    assert _fast_replay_is_complete(cfg, None) is False
    with pytest.raises(ValueError, match="spread calibration"):
        run_backtest_fast(bars, weights, cfg)
    with pytest.raises(ValueError, match="spread calibration"):
        run_backtest(bars, weights, cfg, fast=True)


def test_event_loop_calibrated_spread_at_least_floor() -> None:
    bars, weights = _tiny_ohlc_book()
    flat = _cfg(spread_estimator="flat", half_spread_bps=1.0)
    cal = _cfg(spread_estimator="corwin_schultz", half_spread_bps=1.0)
    r_flat = run_backtest(bars, weights, flat, fast=False)
    r_cal = run_backtest(bars, weights, cal, fast=False)
    assert float(r_cal.metrics["spread"]) >= float(r_flat.metrics["spread"]) - 1e-9
    # Auto-dispatch must not refuse — incompleteness routes to the event loop.
    r_auto = run_backtest(bars, weights, cal, fast=None)
    assert float(r_auto.metrics["spread"]) == pytest.approx(float(r_cal.metrics["spread"]))


def test_pit_map_applies_floor() -> None:
    bars, _ = _tiny_ohlc_book()
    # Tiny lookback with flat prices → undefined / tiny estimate → floor
    flat_bars = bars.with_columns(
        pl.col("close").alias("high"),
        pl.col("close").alias("low"),
    )
    m = pit_calibrated_half_spread_bps(
        flat_bars, estimator="corwin_schultz", lookback=5, floor_bps=7.5
    )
    assert m
    assert all(v == pytest.approx(7.5) for v in m.values())


def test_cost_config_rejects_unknown_estimator() -> None:
    with pytest.raises(ValueError, match="spread_estimator"):
        CostConfig(spread_estimator="kyle")


def test_cost_calibration_trials_and_report(tmp_path: Path) -> None:
    frame, receipt = run_cost_calibration_trials(half_spread_bps=1.0, n_dates=24, n_names=3, seed=3)
    assert receipt["schema"] == COST_CALIBRATION_SCHEMA
    assert receipt["live_pnl_claim"] is False
    assert set(frame["spread_estimator"].to_list()) == {
        "flat",
        "corwin_schultz",
        "abdi_ranaldo",
        "roll",
    }
    flat_spread = float(frame.filter(pl.col("spread_estimator") == "flat")["spread"][0])
    cs_spread = float(frame.filter(pl.col("spread_estimator") == "corwin_schultz")["spread"][0])
    assert cs_spread >= flat_spread - 1e-9
    table = format_cost_calibration_table(frame)
    assert "corwin_schultz" in table
    path = write_cost_calibration_receipt(receipt, tmp_path)
    assert path.exists()
    report = write_cost_calibration_report(frame, receipt, tmp_path / "report.md")
    text = report.read_text(encoding="utf-8")
    assert "flat half-spread" in text
    assert "live_pnl_claim=false" in text.lower()
    assert "corwin_schultz" in text
    assert "Trial results" in text


def test_cli_cost_calibration_dev_gate() -> None:
    from quant_fund.cli.main import app

    denied = runner.invoke(app, ["cost-calibration"])
    assert denied.exit_code != 0
    ok = runner.invoke(app, ["cost-calibration", "--dev", "--n-dates", "20", "--n-names", "3"])
    assert ok.exit_code == 0, ok.output
    assert "corwin_schultz" in ok.output
