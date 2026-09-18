"""microprice_weight_balance fuse propagate + receipt/CLI honesty."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.microstructure.book_panel import write_book_panel
from quant_fund.microstructure.candle_book_features import attach_candle_book_features
from quant_fund.microstructure.synthetic_lob import synthesize_l2_from_bars
from quant_fund.microstructure.vendor_book_map import vendor_panel_from_bars
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import northset_microprice_weight_balance_honesty_errors


def test_attach_propagates_microprice_weight_balance() -> None:
    bars = SyntheticMarketProvider(n_assets=3, n_days=20, seed=2).get_bars()
    book = synthesize_l2_from_bars(bars, depth=5, seed=2)
    assert "microprice_weight_balance" in book.columns
    fused = attach_candle_book_features(bars, book=book, depth=5, seed=2)
    assert "microprice_weight_balance" in fused.columns
    mean = float(fused["microprice_weight_balance"].mean())
    assert math.isfinite(mean)
    assert 0.0 <= mean <= 1.0


def test_candle_book_and_northset_stamp_mean() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars()
    cob = bench_candle_order_book(bars, depth=5, seed=5)
    assert math.isfinite(float(cob["mean_microprice_weight_balance"]))
    assert 0.0 <= float(cob["mean_microprice_weight_balance"]) <= 1.0

    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    receipt = bench_northset(bars, cfg)
    assert math.isfinite(float(receipt["mean_microprice_weight_balance"]))
    assert 0.0 <= float(receipt["mean_microprice_weight_balance"]) <= 1.0
    assert northset_microprice_weight_balance_honesty_errors(receipt) == []


def test_thin_external_missing_col_nan(tmp_path: Path) -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=28, seed=9).get_bars()
    panel = vendor_panel_from_bars(bars, vendor="alpaca", seed=11)
    drop = [c for c in panel.columns if c == "microprice_weight_balance"]
    thin = panel.drop(drop) if drop else panel
    assert "microprice_weight_balance" not in thin.columns
    path = write_book_panel(thin, tmp_path / "thin.parquet")
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    cfg.northset.book_join_coverage_floor = 0.0
    cfg.northset.book_panel_path = str(path)
    receipt = bench_northset(bars, cfg)
    assert math.isnan(float(receipt["mean_microprice_weight_balance"]))
    assert northset_microprice_weight_balance_honesty_errors(receipt) == []


def test_soft_verify_flags_out_of_range() -> None:
    errs = northset_microprice_weight_balance_honesty_errors(
        {"mean_microprice_weight_balance": 1.5}
    )
    assert "mean_microprice_weight_balance_out_of_unit_interval" in errs


def test_northset_cli_echoes_mean() -> None:
    config = Path("configs/research.yaml")
    if not config.exists():
        pytest.skip("configs/research.yaml missing")
    result = CliRunner().invoke(app, ["northset", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "mean_microprice_weight_balance=" in result.output
