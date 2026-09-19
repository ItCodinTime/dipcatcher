"""mean_spread_bps receipt + soft-verify (≈ 2 * mean_half_spread_bps)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import northset_spread_bps_honesty_errors


def _bars():
    return SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars()


def test_candle_book_and_northset_stamp_mean_spread_bps() -> None:
    cob = bench_candle_order_book(_bars(), depth=5, seed=5)
    assert math.isfinite(float(cob["mean_spread_bps"]))
    assert cob["mean_spread_bps"] == pytest.approx(
        2.0 * cob["mean_half_spread_bps"], rel=1e-9, abs=1e-9
    )
    assert northset_spread_bps_honesty_errors(cob) == []

    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    receipt = bench_northset(_bars(), cfg)
    assert math.isfinite(float(receipt["mean_spread_bps"]))
    assert northset_spread_bps_honesty_errors(receipt) == []


def test_soft_verify_flags_bps_not_double_half() -> None:
    errs = northset_spread_bps_honesty_errors(
        {"mean_spread_bps": 10.0, "mean_half_spread_bps": 10.0}
    )
    assert "mean_spread_bps_not_double_mean_half_spread_bps" in errs


def test_cli_echoes_mean_spread_bps() -> None:
    config = Path("configs/research.yaml")
    if not config.exists():
        pytest.skip("configs/research.yaml missing")
    ns = CliRunner().invoke(app, ["northset", "--config", str(config)])
    assert ns.exit_code == 0, ns.output
    assert "mean_spread_bps=" in ns.output
    cb = CliRunner().invoke(app, ["candle-book", "--config", str(config), "--depth", "5"])
    assert cb.exit_code == 0, cb.output
    assert "mean_spread_bps=" in cb.output
