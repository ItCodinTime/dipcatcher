"""E2E: vendor-remapped L2 parquet → Northset book_panel_path honesty."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.book_panel import load_book_panel, write_book_panel
from quant_fund.microstructure.synthetic_lob import synthesize_l2_from_bars
from quant_fund.microstructure.vendor_book_map import (
    disguise_panel_as_alpaca,
    disguise_panel_as_polygon,
    remap_vendor_quotes_to_panel,
    vendor_panel_from_bars,
)
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent


def _bars(n_assets: int = 6, n_days: int = 30, seed: int = 17):
    return SyntheticMarketProvider(n_assets=n_assets, n_days=n_days, seed=seed).get_bars()


def test_vendor_panel_from_bars_stamps_alpaca_source() -> None:
    bars = _bars()
    panel = vendor_panel_from_bars(bars, vendor="alpaca", seed=17)
    assert str(panel["source"][0]) == "alpaca"
    assert panel.height == synthesize_l2_from_bars(bars, seed=17).height


def test_alpaca_disguise_remap_roundtrip_joins_bars(tmp_path: Path) -> None:
    bars = _bars(seed=17)
    raw_panel = synthesize_l2_from_bars(bars, depth=5, seed=17)
    disguised = disguise_panel_as_alpaca(raw_panel)
    raw_path = tmp_path / "alpaca_quotes.parquet"
    disguised.write_parquet(raw_path)
    remapped = remap_vendor_quotes_to_panel(pl.read_parquet(raw_path), vendor="alpaca")
    panel_path = write_book_panel(remapped, tmp_path / "northset_panel.parquet")
    loaded = load_book_panel(panel_path)
    # Join keys must align to bars
    joined = bars.select(["security_id", "event_time"]).join(
        loaded.select(["security_id", "event_time"]),
        on=["security_id", "event_time"],
        how="inner",
    )
    assert joined.height == loaded.height


def test_bench_northset_vendor_book_path_honesty(tmp_path: Path) -> None:
    bars = _bars(n_assets=8, n_days=40, seed=11)
    panel = vendor_panel_from_bars(bars, vendor="alpaca", seed=11)
    path = write_book_panel(panel, tmp_path / "vendor_alpaca_panel.parquet")
    cfg = AppConfig()
    cfg.northset.require_adjusted_ohlc = False
    cfg.data.source = "synthetic"
    cfg.data.synthetic_seed = 11
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False  # isolate book-path honesty
    cfg.northset.book_panel_path = str(path)
    receipt = bench_northset(bars, cfg)
    assert receipt["book_source"] == "alpaca"
    assert receipt["book_dgp"] == "vendor_panel:alpaca"
    assert receipt["dgp"] == "vendor_panel:alpaca"
    assert receipt["book_panel_path"] == str(path)
    assert receipt["data_source"] == "SYNTHETIC"  # bars still synthetic
    assert receipt["label"] == "SYNTHETIC"
    assert receipt["n_fused"] > 0
    assert receipt["research_only"] is True
    assert "live_pnl_claim" not in receipt
    assert family_blob_forbidden_metrics_absent(receipt) is True


def test_bench_northset_polygon_vendor_path(tmp_path: Path) -> None:
    bars = _bars(n_assets=6, n_days=28, seed=9)
    raw = disguise_panel_as_polygon(synthesize_l2_from_bars(bars, seed=9))
    panel = remap_vendor_quotes_to_panel(raw, vendor="polygon")
    path = write_book_panel(panel, tmp_path / "polygon_panel.parquet")
    cfg = AppConfig()
    cfg.northset.require_adjusted_ohlc = False
    cfg.data.source = "synthetic"
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    cfg.northset.book_panel_path = str(path)
    receipt = bench_northset(bars, cfg)
    assert receipt["book_source"] == "polygon"
    assert receipt["dgp"] == "vendor_panel:polygon"


def test_load_book_panel_missing_path_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_book_panel(tmp_path / "does_not_exist.parquet")


def test_write_alpaca_fixture_pack(tmp_path: Path) -> None:
    """Materialize a small fixture pack (also usable under tests/fixtures)."""
    bars = _bars(n_assets=4, n_days=20, seed=3)
    raw_panel = synthesize_l2_from_bars(bars, seed=3)
    alpaca = disguise_panel_as_alpaca(raw_panel)
    out = tmp_path / "alpaca_quotes.parquet"
    alpaca.write_parquet(out)
    remapped = remap_vendor_quotes_to_panel(pl.read_parquet(out), vendor="alpaca")
    assert remapped.height == alpaca.height
