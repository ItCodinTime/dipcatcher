"""Vendor book-map edge branches: unmapped/alias/dedup paths + PIT clocks."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.vendor_book_map import (
    _apply_sip_participant_clocks,
    _clock_labels,
    disguise_panel_as_alpaca,
    disguise_panel_as_polygon,
    dry_run_vendor_bar_map,
    dry_run_vendor_book_map,
    remap_vendor_bars,
    remap_vendor_quotes_to_panel,
    vendor_panel_from_bars,
)


def _generic_quotes(**overrides: object) -> pl.DataFrame:
    """Minimal generic-vendor quote frame (all top-of-book targets mappable)."""
    t0 = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    rows: dict[str, list[object]] = {
        "security_id": ["SEC_1", "SEC_1"],
        "timestamp": [t0, t0 + timedelta(minutes=1)],
        "bid": [99.99, 100.01],
        "ask": [100.01, 100.05],
        "bid_size": [100.0, 90.0],
        "ask_size": [80.0, 95.0],
    }
    for col, values in overrides.items():
        rows[col] = list(values)  # type: ignore[call-overload]
    return pl.DataFrame(rows)


def _generic_bars(**overrides: object) -> pl.DataFrame:
    t0 = datetime(2024, 1, 2, 20, 0, tzinfo=UTC)
    rows: dict[str, list[object]] = {
        "security_id": ["SEC_1", "SEC_1"],
        "timestamp": [t0, t0 + timedelta(days=1)],
        "open": [100.0, 101.0],
        "high": [101.0, 102.0],
        "low": [99.0, 100.0],
        "close": [100.5, 101.5],
        "volume": [1e6, 1.1e6],
    }
    for col, values in overrides.items():
        rows[col] = list(values)  # type: ignore[call-overload]
    return pl.DataFrame(rows)


def test_dry_run_reports_missing_targets_unused_columns_and_dict() -> None:
    cols = ["S", "t", "bp", "ap", "zz_extra", "volume", "aa_other"]
    report = dry_run_vendor_book_map(cols, vendor="generic")
    assert report.derivable_ok is False
    assert report.missing_targets == ("top_bid_size", "top_ask_size")
    assert report.unused_vendor_columns == ("aa_other", "volume", "zz_extra")
    assert report.mapped["best_bid"] == "bp"
    assert any("incomplete top-of-book" in n for n in report.notes)
    assert any("source missing" in n for n in report.notes)
    blob = report.as_dict()
    assert blob["vendor"] == "generic"
    assert blob["research_only"] is True
    assert blob["claim"] == "research_diagnostic_only"
    assert blob["missing_targets"] == ["top_bid_size", "top_ask_size"]
    assert blob["unused_vendor_columns"] == ["aa_other", "volume", "zz_extra"]


def test_dry_run_rejects_unknown_vendor_preset() -> None:
    with pytest.raises(ValueError, match="unknown vendor preset"):
        dry_run_vendor_book_map(["S", "t"], vendor="bloomberg")
    with pytest.raises(ValueError, match="unknown vendor bar preset"):
        dry_run_vendor_bar_map(["ticker", "t"], vendor="bats")


def test_alias_first_match_wins_and_reports_rest_unused() -> None:
    cols = ["security_id", "timestamp", "bid_price", "bid", "ask", "bid_size", "ask_size"]
    report = dry_run_vendor_book_map(cols, vendor="generic")
    # bid_price precedes bid in the generic alias tuple.
    assert report.mapped["best_bid"] == "bid_price"
    assert "bid" in report.unused_vendor_columns
    report = dry_run_vendor_book_map(
        ["S", "t", "bp", "bid_price", "ap", "bs", "as"], vendor="alpaca"
    )
    # bp precedes bid_price in the alpaca alias tuple.
    assert report.mapped["best_bid"] == "bp"
    assert "bid_price" in report.unused_vendor_columns


def test_one_vendor_column_dedupes_onto_both_clock_targets() -> None:
    t0 = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    raw = pl.DataFrame(
        {
            "S": ["SEC_1", "SEC_1"],
            "t": [t0, t0 + timedelta(minutes=1)],
            "bp": [99.99, 100.01],
            "ap": [100.01, 100.05],
            "bs": [100.0, 90.0],
            "as": [80.0, 95.0],
        }
    )
    report = dry_run_vendor_book_map(list(raw.columns), vendor="alpaca")
    # A single vendor column fans out to event_time and available_time.
    assert report.mapped["event_time"] == report.mapped["available_time"] == "t"
    panel = remap_vendor_quotes_to_panel(raw, vendor="alpaca")
    assert (panel["event_time"] == raw["t"]).all()
    assert (panel["available_time"] == raw["t"]).all()


def test_clock_skew_takes_max_of_sip_and_participant() -> None:
    t0 = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    rows = []
    for i, lag_ms in enumerate((200, -50)):
        venue = t0 + timedelta(milliseconds=i)
        sip = venue + timedelta(milliseconds=lag_ms)
        mid = 100.0 + i
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
    panel = remap_vendor_quotes_to_panel(raw, vendor="polygon")
    # Clock skew can never mint available_time < event_time.
    assert (panel["available_time"] >= panel["event_time"]).all()
    by_event = panel.sort("event_time")
    # Row 0: sip lags participant (normal) -> available = sip.
    assert by_event["available_time"][0] == raw["sip_timestamp"][0]
    # Row 1: sip precedes participant (skew) -> available = participant.
    assert by_event["available_time"][1] == raw["participant_timestamp"][1]


def test_participant_only_quotes_stamp_available_clock_unverified() -> None:
    t0 = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    raw = _generic_quotes()
    raw = raw.with_columns(
        pl.Series("participant_timestamp", [t0, t0 + timedelta(minutes=1)]),
        (pl.col("timestamp") + timedelta(milliseconds=300)).alias("timestamp"),
    )
    report = dry_run_vendor_book_map(list(raw.columns), vendor="generic")
    assert report.derivable_ok is True
    assert report.event_clock == "participant"
    assert report.available_clock == "vendor_timestamp"
    panel = remap_vendor_quotes_to_panel(raw, vendor="generic")
    # Remap restamps the availability clock: venue time without SIP is unverified.
    assert (panel["event_time"] == raw["participant_timestamp"]).all()
    assert (panel["available_time"] == raw["timestamp"]).all()
    assert str(panel["event_clock"][0]) == "participant"
    assert str(panel["available_clock"][0]) == "participant_unverified"


def test_clock_labels_participant_mapped_available_is_unverified() -> None:
    # No shipped preset maps available_time onto participant_timestamp; the
    # helper still resolves the combination honestly when a map does.
    clocks = _clock_labels(
        {"participant_timestamp", "bid", "ask"},
        {"event_time": "participant_timestamp", "available_time": "participant_timestamp"},
    )
    assert clocks == ("participant", "participant_unverified")


def test_apply_clocks_rejects_misaligned_clock_columns() -> None:
    t0 = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    original = pl.DataFrame(
        {
            "participant_timestamp": [t0 + timedelta(milliseconds=i) for i in range(3)],
            "sip_timestamp": [t0 + timedelta(milliseconds=i + 100) for i in range(3)],
        }
    )
    base = pl.DataFrame({"security_id": ["A", "B"]})
    with pytest.raises(ValueError, match="clock columns do not align"):
        _apply_sip_participant_clocks(original, base)


def test_vendor_source_column_survives_remap() -> None:
    raw = _generic_quotes(source=["xnas", "xnas"])
    report = dry_run_vendor_book_map(list(raw.columns), vendor="generic")
    assert report.mapped["source"] == "source"
    assert not any("source missing" in n for n in report.notes)
    panel = remap_vendor_quotes_to_panel(raw, vendor="generic")
    assert (panel["source"] == "xnas").all()


def test_remap_quotes_default_source_names_the_frame() -> None:
    t0 = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    raw = pl.DataFrame(
        {
            "S": ["SEC_1"],
            "t": [t0],
            "bp": [99.99],
            "ap": [100.01],
            "bs": [100.0],
            "as": [80.0],
        }
    )
    panel = remap_vendor_quotes_to_panel(raw, vendor="alpaca", default_source="research-feed")
    assert panel["source"][0] == "research-feed"


def test_zero_top_sizes_fail_closed_at_validation() -> None:
    raw = _generic_quotes(bid_size=[0.0, 0.0], ask_size=[0.0, 0.0])
    with pytest.raises(ValueError, match="non-positive"):
        remap_vendor_quotes_to_panel(raw, vendor="generic")


def test_disguise_helpers_fail_closed_on_missing_columns() -> None:
    thin = pl.DataFrame(
        {
            "security_id": ["SEC_1"],
            "event_time": [datetime(2024, 1, 2, tzinfo=UTC)],
            "best_bid": [99.99],
        }
    )
    with pytest.raises(ValueError, match="panel missing columns"):
        disguise_panel_as_alpaca(thin)
    with pytest.raises(ValueError, match="panel missing columns"):
        disguise_panel_as_polygon(thin)


def test_vendor_panel_from_bars_disguises_then_remaps() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    panel = vendor_panel_from_bars(bars, vendor=" Polygon ")
    assert panel.height > 0
    assert (panel["source"] == "polygon").all()
    alpaca_panel = vendor_panel_from_bars(bars)
    assert alpaca_panel.height > 0
    assert (alpaca_panel["source"] == "alpaca").all()
    with pytest.raises(ValueError, match=r"supports alpaca\|polygon"):
        vendor_panel_from_bars(bars, vendor="generic")


def test_bars_keep_vendor_source_column_and_default_source() -> None:
    raw = _generic_bars(exchange=["XNYS", "XNYS"])
    bars = remap_vendor_bars(raw, vendor="generic")
    assert (bars["source"] == "XNYS").all()
    assert str(bars["event_clock"][0]) == "vendor_timestamp"
    named = remap_vendor_bars(_generic_bars(), vendor="generic", default_source="research-feed")
    assert (named["source"] == "research-feed").all()


def test_bars_participant_only_fail_closed_without_sip() -> None:
    t0 = datetime(2024, 1, 2, 20, 0, tzinfo=UTC)
    raw = _generic_bars(
        participant_timestamp=[t0, t0 + timedelta(days=1)],
        timestamp=[t0 + timedelta(milliseconds=400), t0 + timedelta(days=1, milliseconds=400)],
    )
    report = dry_run_vendor_bar_map(list(raw.columns), vendor="generic")
    assert report.derivable_ok is True
    assert report.event_clock == "participant"
    with pytest.raises(ValueError, match="participant-only timestamps"):
        remap_vendor_bars(raw, vendor="generic")


def test_bars_available_time_before_event_time_fail_closed() -> None:
    t0 = datetime(2024, 1, 2, 20, 0, tzinfo=UTC)
    raw = _generic_bars(
        event_time=[t0, t0 + timedelta(days=1)],
        timestamp=[t0 - timedelta(milliseconds=1), t0 + timedelta(days=1)],
    )
    report = dry_run_vendor_bar_map(list(raw.columns), vendor="generic")
    assert report.derivable_ok is True
    with pytest.raises(ValueError, match="available_time before event_time"):
        remap_vendor_bars(raw, vendor="generic")


def test_dry_run_bar_map_reports_missing_volume_and_unused() -> None:
    cols = ["ticker", "timestamp", "o", "h", "l", "c", "zz_extra"]
    report = dry_run_vendor_bar_map(cols, vendor="generic")
    assert report.derivable_ok is False
    assert report.missing_targets == ("volume",)
    assert report.unused_vendor_columns == ("zz_extra",)
    assert any("incomplete OHLCV" in n for n in report.notes)
