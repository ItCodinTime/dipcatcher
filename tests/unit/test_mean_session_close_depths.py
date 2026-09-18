"""mean_session_close_bid/ask_depth receipt + CLI; ≥0 when finite."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset


def test_stamps_mean_session_close_depths_when_session_l2_on() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=23).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = True
    receipt = bench_northset(bars, cfg)
    bid = float(receipt["mean_session_close_bid_depth"])
    ask = float(receipt["mean_session_close_ask_depth"])
    assert math.isfinite(bid) and math.isfinite(ask)
    assert bid >= 0.0 and ask >= 0.0


def test_nan_when_session_l2_off() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=23).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    receipt = bench_northset(bars, cfg)
    assert math.isnan(float(receipt["mean_session_close_bid_depth"]))
    assert math.isnan(float(receipt["mean_session_close_ask_depth"]))


def test_northset_cli_echoes_mean_session_close_depths() -> None:
    config = Path("configs/research.yaml")
    if not config.exists():
        pytest.skip("configs/research.yaml missing")
    result = CliRunner().invoke(app, ["northset", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "mean_session_close_bid_depth=" in result.output
    assert "mean_session_close_ask_depth=" in result.output
