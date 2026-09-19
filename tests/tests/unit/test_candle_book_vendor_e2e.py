"""Vendor-remapped L2 → candle fuse → date-IC bench (offline fixtures)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.microstructure.book_panel import validate_book_panel
from quant_fund.microstructure.candle_book_features import attach_candle_book_features
from quant_fund.microstructure.vendor_book_map import remap_vendor_quotes_to_panel
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent

_FIX = Path(__file__).resolve().parents[1] / "fixtures" / "northset"


def test_vendor_panel_fuse_bench_e2e() -> None:
    bars = pl.read_parquet(_FIX / "bars_aligned.parquet")
    panel = validate_book_panel(pl.read_parquet(_FIX / "alpaca_remapped_panel.parquet"))
    assert str(panel["source"][0]) == "alpaca"
    fused = attach_candle_book_features(bars, book=panel)
    assert fused["book_source"][0] == "alpaca"
    assert fused["book_dgp"][0] == "external_panel"
    receipt = bench_candle_order_book(bars, book=panel, min_names=2, label="SYNTHETIC")
    assert receipt["book_source"] == "alpaca"
    assert receipt["book_dgp"] == "external_panel"
    assert receipt["ic_method"] == "date_level_spearman_hac"
    assert receipt["n_scored"] > 0
    assert family_blob_forbidden_metrics_absent(receipt) is True


def test_raw_alpaca_quotes_remap_then_bench() -> None:
    bars = pl.read_parquet(_FIX / "bars_aligned.parquet")
    raw = pl.read_parquet(_FIX / "alpaca_quotes.parquet")
    panel = remap_vendor_quotes_to_panel(raw, vendor="alpaca")
    receipt = bench_candle_order_book(bars, book=panel, min_names=2, label="SYNTHETIC")
    assert receipt["book_source"] == "alpaca"
    assert "ic_imbalance_top" in receipt


def test_vendor_top_of_book_derives_ofi_proxy() -> None:
    bars = pl.read_parquet(_FIX / "bars_aligned.parquet")
    panel = validate_book_panel(pl.read_parquet(_FIX / "alpaca_remapped_panel.parquet"))
    assert "ofi" not in panel.columns
    fused = attach_candle_book_features(bars, book=panel)
    assert "ofi" in fused.columns
    assert "queue_imbalance" in fused.columns
    receipt = bench_candle_order_book(bars, book=panel, min_names=2, label="SYNTHETIC")
    assert "ic_ofi" in receipt
