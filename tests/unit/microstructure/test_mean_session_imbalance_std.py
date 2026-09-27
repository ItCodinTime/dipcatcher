"""mean_session_imbalance_std receipt + CLI; path dispersion ≥0; receipt-only."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset


def test_stamps_mean_session_imbalance_std_when_session_l2_on() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=29).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = True
    receipt = bench_northset(bars, cfg)
    std_m = float(receipt["mean_session_imbalance_std"])
    path_m = float(receipt["mean_session_imbalance_mean"])
    close_m = float(receipt["mean_session_close_imbalance"])
    assert math.isfinite(std_m) and std_m >= 0.0
    assert math.isfinite(path_m) and math.isfinite(close_m)
    # Distinct receipt keys (dispersion ≠ mean ≠ last-snap)
    assert "mean_session_imbalance_std" in receipt
    assert "mean_session_imbalance_mean" in receipt
    assert "mean_session_close_imbalance" in receipt


def test_nan_when_session_l2_off() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=29).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    receipt = bench_northset(bars, cfg)
    assert math.isnan(float(receipt["mean_session_imbalance_std"]))


def test_northset_cli_echoes_mean_session_imbalance_std() -> None:
    config = Path("configs/research.yaml")
    if not config.exists():
        pytest.skip("configs/research.yaml missing")
    result = CliRunner().invoke(app, ["northset", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "mean_session_imbalance_std=" in result.output
