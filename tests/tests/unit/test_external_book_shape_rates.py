"""External book_panel_path: honest NaN shape rates; floors fail-closed."""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.book_panel import write_book_panel
from quant_fund.microstructure.vendor_book_map import vendor_panel_from_bars
from quant_fund.northset.benches import bench_northset


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False  # isolate daily external book
    cfg.northset.book_join_coverage_floor = 0.0  # allow any join for fixture size
    return cfg


def _tob_panel_without_shape(bars: pl.DataFrame, path: Path) -> Path:
    """Vendor-shaped top-of-book panel: strip n_*_levels + DEPTH_SHAPE/SIDE_STRUCTURE."""
    panel = vendor_panel_from_bars(bars, vendor="alpaca", seed=11)
    drop = [
        c
        for c in panel.columns
        if c.startswith("n_")
        or "log_price_slope" in c
        or "log_size_slope" in c
        or "tick_spacing" in c
        or "concentration" in c
    ]
    # force-drop n_* even if named differently
    for c in ("n_bid_levels", "n_ask_levels"):
        if c in panel.columns and c not in drop:
            drop.append(c)
    thin = panel.drop(drop) if drop else panel
    assert "n_bid_levels" not in thin.columns
    assert "bid_log_price_slope" not in thin.columns
    return write_book_panel(thin, path)


def test_external_panel_missing_shape_cols_stamps_nan(tmp_path: Path) -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=28, seed=9).get_bars()
    path = _tob_panel_without_shape(bars, tmp_path / "tob_no_shape.parquet")
    cfg = _cfg()
    cfg.northset.book_panel_path = str(path)
    cfg.northset.depth_shape_finite_floor = None
    cfg.northset.concentration_top_finite_floor = None
    receipt = bench_northset(bars, cfg)
    assert receipt["shape_columns_ensured"] is False
    assert math.isnan(float(receipt["depth_shape_finite_rate"]))
    assert math.isnan(float(receipt["concentration_top_finite_rate"]))
    assert receipt["depth_shape_finite_floor"] is None
    assert receipt["concentration_top_finite_floor"] is None


def test_external_panel_floors_fail_closed_on_nan(tmp_path: Path) -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=28, seed=9).get_bars()
    path = _tob_panel_without_shape(bars, tmp_path / "tob_no_shape_floor.parquet")
    cfg = _cfg()
    cfg.northset.book_panel_path = str(path)
    cfg.northset.depth_shape_finite_floor = 0.5
    with pytest.raises(ValueError, match="depth_shape_finite_rate"):
        bench_northset(bars, cfg)

    cfg2 = _cfg()
    cfg2.northset.book_panel_path = str(path)
    cfg2.northset.concentration_top_finite_floor = 0.5
    with pytest.raises(ValueError, match="concentration_top_finite_rate"):
        bench_northset(bars, cfg2)
