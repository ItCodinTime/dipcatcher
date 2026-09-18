"""mean_session_close_spread_bps receipt + CLI; ≠ path / daily spread means."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset


def test_stamps_mean_session_close_spread_bps_when_session_l2_on() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=11).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = True
    receipt = bench_northset(bars, cfg)
    close_m = float(receipt["mean_session_close_spread_bps"])
    path_m = float(receipt["mean_session_spread_bps_mean"])
    daily_m = float(receipt["mean_spread_bps"])
    assert math.isfinite(close_m)
    assert math.isfinite(path_m)
    assert math.isfinite(daily_m)
    # Honesty: three distinct receipt keys (last-snap / path / daily) — values
    # may coincide by chance on tiny synthetic books; do not equate identities.
    assert "mean_session_close_spread_bps" in receipt
    assert "mean_session_spread_bps_mean" in receipt
    assert "mean_spread_bps" in receipt


def test_nan_when_session_l2_off() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=11).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    receipt = bench_northset(bars, cfg)
    assert math.isnan(float(receipt["mean_session_close_spread_bps"]))


def test_northset_cli_echoes_mean_session_close_spread_bps() -> None:
    config = Path("configs/research.yaml")
    if not config.exists():
        pytest.skip("configs/research.yaml missing")
    result = CliRunner().invoke(app, ["northset", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "mean_session_close_spread_bps=" in result.output
