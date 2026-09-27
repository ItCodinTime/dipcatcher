"""Candle-book + northset spread-alias receipt parity."""

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
from quant_fund.research.catalog import (
    northset_half_spread_honesty_errors,
    northset_spread_receipt_honesty_errors,
)


def _bars():
    return SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars()


def test_candle_book_stamps_spread_alias_means() -> None:
    receipt = bench_candle_order_book(_bars(), depth=5, seed=5)
    for key in (
        "mean_quoted_spread",
        "mean_effective_spread",
        "mean_half_spread",
        "mean_half_spread_bps",
    ):
        assert math.isfinite(float(receipt[key])), key
    assert receipt["mean_effective_spread"] == pytest.approx(
        receipt["mean_quoted_spread"], rel=1e-9, abs=1e-9
    )
    assert receipt["mean_half_spread"] == pytest.approx(
        0.5 * receipt["mean_quoted_spread"], rel=1e-9, abs=1e-9
    )
    assert northset_spread_receipt_honesty_errors(receipt) == []
    assert northset_half_spread_honesty_errors(receipt) == []


def test_northset_stamps_half_spread_means() -> None:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    receipt = bench_northset(_bars(), cfg)
    assert math.isfinite(float(receipt["mean_half_spread"]))
    assert math.isfinite(float(receipt["mean_half_spread_bps"]))
    assert northset_half_spread_honesty_errors(receipt) == []


def test_soft_verify_flags_half_not_half() -> None:
    errs = northset_half_spread_honesty_errors({"mean_quoted_spread": 0.2, "mean_half_spread": 0.2})
    assert "mean_half_spread_not_half_of_mean_quoted_spread" in errs


def test_northset_cli_echoes_half_means() -> None:
    config = Path("configs/research.yaml")
    if not config.exists():
        pytest.skip("configs/research.yaml missing")
    result = CliRunner().invoke(app, ["northset", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "mean_half_spread=" in result.output
    assert "mean_half_spread_bps=" in result.output
    assert "mean_close_mid_abs_rel=" in result.output


def test_candle_book_cli_echoes_spread_means(tmp_path: Path) -> None:
    config = Path("configs/research.yaml")
    if not config.exists():
        pytest.skip("configs/research.yaml missing")
    result = CliRunner().invoke(app, ["candle-book", "--config", str(config), "--depth", "5"])
    assert result.exit_code == 0, result.output
    assert "mean_quoted_spread=" in result.output
    assert "mean_half_spread=" in result.output
