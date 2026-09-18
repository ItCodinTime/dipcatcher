"""Northset join_coverage stamps + optional kyle_ofi nest."""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.book_panel import write_book_panel
from quant_fund.microstructure.synthetic_lob import synthesize_l2_from_bars
from quant_fund.microstructure.vendor_book_map import vendor_panel_from_bars
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent


def _bars(n_assets: int = 6, n_days: int = 30, seed: int = 21):
    return SyntheticMarketProvider(n_assets=n_assets, n_days=n_days, seed=seed).get_bars()


def test_receipt_stamps_join_coverage_and_book_age() -> None:
    bars = _bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    receipt = bench_northset(bars, cfg)
    assert "join_coverage" in receipt
    assert receipt["join_coverage"] == receipt["join_coverage"]  # finite
    assert receipt["join_coverage"] >= 0.99
    assert "mean_book_age_seconds" in receipt
    assert "max_book_age_seconds" in receipt
    assert receipt["include_kyle_ofi"] is False
    assert "kyle_ofi" not in receipt


def test_external_panel_fail_closed_on_low_join_coverage(tmp_path: Path) -> None:
    bars = _bars(n_assets=8, n_days=40, seed=11)
    # Tiny misaligned panel → low coverage
    tiny = bars.head(3)
    panel = synthesize_l2_from_bars(tiny, seed=11)
    path = write_book_panel(panel, tmp_path / "tiny.parquet")
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    cfg.northset.book_panel_path = str(path)
    cfg.northset.book_join_coverage_floor = 0.5
    with pytest.raises(ValueError, match="join_coverage|min_join_coverage"):
        bench_northset(bars, cfg)


def test_vendor_panel_passes_join_floor_and_stamps(tmp_path: Path) -> None:
    bars = _bars(n_assets=6, n_days=28, seed=9)
    panel = vendor_panel_from_bars(bars, vendor="alpaca", seed=9)
    path = write_book_panel(panel, tmp_path / "alpaca.parquet")
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    cfg.northset.book_panel_path = str(path)
    cfg.northset.book_join_coverage_floor = 0.5
    receipt = bench_northset(bars, cfg)
    assert receipt["join_coverage"] >= 0.5
    assert receipt["book_join_coverage_floor"] == 0.5
    assert receipt["book_source"] == "alpaca"


def test_include_kyle_ofi_nests_research_only_blob() -> None:
    bars = _bars(n_assets=6, n_days=30, seed=7)
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    cfg.northset.include_kyle_ofi = True
    receipt = bench_northset(bars, cfg)
    assert receipt["include_kyle_ofi"] is True
    nest = receipt["kyle_ofi"]
    assert isinstance(nest, dict)
    assert nest.get("family") == "kyle_ofi"
    assert nest.get("research_only") is True
    assert "live_pnl_claim" not in nest
    assert "kyle_lambda_depth_mean" in nest or "kyle_lambda_ofi_mean" in nest
    assert family_blob_forbidden_metrics_absent(receipt) is True
