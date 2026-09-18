"""mean_session_close_micro_bps receipt + CLI; ≠ daily microprice mean."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import northset_session_close_micro_bps_honesty_errors


def test_stamps_mean_session_close_micro_bps_when_session_l2_on() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=17).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = True
    receipt = bench_northset(bars, cfg)
    val = float(receipt["mean_session_close_micro_bps"])
    assert math.isfinite(val)
    assert northset_session_close_micro_bps_honesty_errors(receipt) == []


def test_nan_when_session_l2_off() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=17).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    receipt = bench_northset(bars, cfg)
    assert math.isnan(float(receipt["mean_session_close_micro_bps"]))
    assert northset_session_close_micro_bps_honesty_errors(receipt) == []


def test_northset_cli_echoes_mean_session_close_micro_bps() -> None:
    config = Path("configs/research.yaml")
    if not config.exists():
        pytest.skip("configs/research.yaml missing")
    result = CliRunner().invoke(app, ["northset", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "mean_session_close_micro_bps=" in result.output
