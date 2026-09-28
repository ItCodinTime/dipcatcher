"""Unit tests for Kyle/OFI fuse helpers extracted for the McCabe ratchet."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_fund.northset.kyle_ofi import (
    _attach_ofi_if_missing,
    _ensure_depth_aliases,
    _forward_target_exprs,
    _join_bars_book_panel,
    _prefer_close_forward_returns,
    _stamp_book_honesty,
    fuse_bars_l2_kyle_frame,
)


def _bars(n: int = 4) -> pl.DataFrame:
    start = datetime(2024, 1, 2, tzinfo=UTC)
    return pl.DataFrame(
        {
            "security_id": ["A"] * n,
            "event_time": [start + timedelta(days=i) for i in range(n)],
            "mid": [100.0 + i for i in range(n)],
            "close": [100.0 + i for i in range(n)],
            "best_bid": [99.5 + i for i in range(n)],
            "best_ask": [100.5 + i for i in range(n)],
            "top_bid_size": [10.0] * n,
            "top_ask_size": [12.0] * n,
        }
    )


def test_stamp_book_honesty_synthetic_and_vendor() -> None:
    assert _stamp_book_honesty(pl.DataFrame({"x": [1]}), synthesized=True) == (
        "synthetic_lob",
        "synthetic_lob",
    )
    vendor = pl.DataFrame({"source": ["vendor_x", "vendor_x"]})
    assert _stamp_book_honesty(vendor, synthesized=False) == (
        "vendor_x",
        "vendor_panel:vendor_x",
    )
    with pytest.raises(ValueError, match="missing required 'source'"):
        _stamp_book_honesty(pl.DataFrame({"x": [1]}), synthesized=False)


def test_join_bars_book_panel_passthrough_and_join() -> None:
    bars = _bars()
    frame, source, dgp, synthesized = _join_bars_book_panel(bars, None)
    assert synthesized is True
    assert source == "synthetic_lob"
    assert frame.height == bars.height
    book = bars.with_columns(pl.lit("vendor_x").alias("source"))
    fused, source, dgp, synthesized = _join_bars_book_panel(bars, book)
    assert synthesized is False
    assert source == "vendor_x"
    assert dgp == "vendor_panel:vendor_x"
    assert fused.height == bars.height


def test_ensure_depth_aliases_and_forward_exprs() -> None:
    frame = _ensure_depth_aliases(_bars().drop(["bid_depth", "ask_depth"], strict=False))
    assert "bid_depth" in frame.columns and "ask_depth" in frame.columns
    exprs = _forward_target_exprs(
        mid_col="mid",
        book_source="synthetic_lob",
        book_dgp="synthetic_lob",
        join_coverage=1.0,
        has_close=True,
    )
    assert len(exprs) == 12  # 9 base+honesty + 3 close fwd
    built = frame.sort(["security_id", "event_time"]).with_columns(exprs)
    built = _prefer_close_forward_returns(built)
    assert "fwd_ret_1" in built.columns
    with_ofi = _attach_ofi_if_missing(built.drop("ofi", strict=False))
    assert "ofi" in with_ofi.columns


def test_fuse_bars_l2_kyle_frame_smoke() -> None:
    out = fuse_bars_l2_kyle_frame(_bars())
    assert out.height == 4
    assert {"signed_depth", "ofi", "delta_mid", "fwd_ret_1", "join_coverage"} <= set(out.columns)
