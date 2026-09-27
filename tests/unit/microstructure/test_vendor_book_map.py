"""Vendor book column-map dry-run + remap (offline)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from quant_fund.microstructure.book_panel import BOOK_PANEL_REQUIRED, validate_book_panel
from quant_fund.microstructure.vendor_book_map import (
    dry_run_vendor_bar_map,
    dry_run_vendor_book_map,
    remap_vendor_bars,
    remap_vendor_quotes_to_panel,
)


def _alpaca_shaped(n: int = 40) -> pl.DataFrame:
    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    rows = []
    for i in range(n):
        mid = 100.0 + 0.01 * i
        rows.append(
            {
                "S": f"SEC_{(i % 5) + 1:04d}",
                "t": start + timedelta(minutes=i),
                "bp": mid - 0.01,
                "ap": mid + 0.01,
                "bs": 100.0 + i,
                "as": 90.0 + i,
            }
        )
    return pl.DataFrame(rows)


def test_alpaca_dry_run_complete() -> None:
    report = dry_run_vendor_book_map(["S", "t", "bp", "ap", "bs", "as"], vendor="alpaca")
    assert report.derivable_ok is True
    assert report.missing_targets == ()
    assert report.mapped["best_bid"] == "bp"


def test_polygon_dry_run_complete() -> None:
    report = dry_run_vendor_book_map(
        ["ticker", "sip_timestamp", "bid", "ask", "bid_size", "ask_size"],
        vendor="polygon",
    )
    assert report.derivable_ok is True
    assert report.event_clock == "sip"
    assert report.available_clock == "sip"


def test_polygon_participant_and_sip_are_distinct_pit_clocks() -> None:
    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    rows = []
    for i in range(8):
        mid = 100.0 + 0.01 * i
        venue = start + timedelta(milliseconds=i)
        sip = venue + timedelta(milliseconds=200)
        rows.append(
            {
                "ticker": "AAPL",
                "participant_timestamp": venue,
                "sip_timestamp": sip,
                "bid": mid - 0.01,
                "ask": mid + 0.01,
                "bid_size": 100.0,
                "ask_size": 90.0,
            }
        )
    raw = pl.DataFrame(rows)
    report = dry_run_vendor_book_map(list(raw.columns), vendor="polygon")
    assert report.event_clock == "participant"
    assert report.available_clock == "sip"
    panel = remap_vendor_quotes_to_panel(raw, vendor="polygon")
    validate_book_panel(panel)
    assert (panel["event_time"] == raw["participant_timestamp"]).all()
    assert (panel["available_time"] == raw["sip_timestamp"]).all()
    assert (panel["available_time"] >= panel["event_time"]).all()
    assert str(panel["event_clock"][0]) == "participant"
    assert str(panel["available_clock"][0]) == "sip"


def test_polygon_participant_only_fails_closed_without_sip() -> None:
    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    raw = pl.DataFrame(
        {
            "ticker": ["AAPL"],
            "participant_timestamp": [start],
            "bid": [99.9],
            "ask": [100.1],
            "bid_size": [10.0],
            "ask_size": [10.0],
        }
    )
    report = dry_run_vendor_book_map(list(raw.columns), vendor="polygon")
    assert report.derivable_ok is False
    assert "available_time" in report.missing_targets
    with pytest.raises(ValueError, match="incomplete"):
        remap_vendor_quotes_to_panel(raw, vendor="polygon")


def test_dry_run_flags_missing() -> None:
    report = dry_run_vendor_book_map(["S", "t", "bp"], vendor="alpaca")
    assert report.derivable_ok is False
    assert "best_ask" in report.missing_targets


def test_remap_alpaca_to_validated_panel() -> None:
    raw = _alpaca_shaped()
    panel = remap_vendor_quotes_to_panel(raw, vendor="alpaca")
    validate_book_panel(panel)
    for col in BOOK_PANEL_REQUIRED:
        assert col in panel.columns
    assert panel.height == raw.height
    assert (panel["best_bid"] < panel["best_ask"]).all()
    assert str(panel["source"][0]) == "alpaca"


def test_remap_rejects_incomplete() -> None:
    raw = _alpaca_shaped().drop("ap")
    with pytest.raises(ValueError, match="incomplete"):
        remap_vendor_quotes_to_panel(raw, vendor="alpaca")


def test_remap_roundtrip_parquet(tmp_path: Path) -> None:
    raw = _alpaca_shaped()
    path = tmp_path / "alpaca_quotes.parquet"
    raw.write_parquet(path)
    panel = remap_vendor_quotes_to_panel(pl.read_parquet(path), vendor="alpaca")
    out = tmp_path / "panel.parquet"
    panel.write_parquet(out)
    assert pl.read_parquet(out).height == raw.height


def test_polygon_bars_participant_and_sip_are_distinct_pit_clocks() -> None:
    start = datetime(2024, 1, 2, 20, 0, tzinfo=UTC)
    rows = []
    for i in range(6):
        venue = start + timedelta(days=i)
        sip = venue + timedelta(milliseconds=400)
        px = 100.0 + i
        rows.append(
            {
                "ticker": "AAPL",
                "participant_timestamp": venue,
                "sip_timestamp": sip,
                "o": px,
                "h": px + 1.0,
                "l": px - 1.0,
                "c": px + 0.2,
                "v": 1_000_000.0 + i,
            }
        )
    raw = pl.DataFrame(rows)
    report = dry_run_vendor_bar_map(list(raw.columns), vendor="polygon")
    assert report.derivable_ok is True
    assert report.event_clock == "participant"
    assert report.available_clock == "sip"
    bars = remap_vendor_bars(raw, vendor="polygon")
    assert (bars["event_time"] == raw["participant_timestamp"]).all()
    assert (bars["available_time"] == raw["sip_timestamp"]).all()
    assert (bars["available_time"] >= bars["event_time"]).all()
    assert str(bars["event_clock"][0]) == "participant"
    assert str(bars["available_clock"][0]) == "sip"
    assert str(bars["source"][0]) == "polygon"


def test_polygon_bars_participant_only_fails_closed_without_sip() -> None:
    start = datetime(2024, 1, 2, 20, 0, tzinfo=UTC)
    raw = pl.DataFrame(
        {
            "ticker": ["AAPL"],
            "participant_timestamp": [start],
            "o": [100.0],
            "h": [101.0],
            "l": [99.0],
            "c": [100.5],
            "v": [1e6],
        }
    )
    report = dry_run_vendor_bar_map(list(raw.columns), vendor="polygon")
    assert report.derivable_ok is False
    assert "available_time" in report.missing_targets
    with pytest.raises(ValueError, match="incomplete"):
        remap_vendor_bars(raw, vendor="polygon")
