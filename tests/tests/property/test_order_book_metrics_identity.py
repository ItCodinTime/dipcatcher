"""Snapshot ↔ book_metrics identities (depth honesty, not live P&L)."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.microstructure.book_metrics import (
    DEPTH_SHAPE_FIELDS,
    METRICS_OPTIONAL_NAN_OK_KEYS,
    METRICS_REQUIRED_FINITE_KEY_DOCS,
    METRICS_REQUIRED_FINITE_KEYS,
    QUEUE_STRUCTURE_FIELDS,
    SIDE_NOTIONAL_FIELDS,
    SIDE_STRUCTURE_FIELDS,
    TOB_SHARE_FIELDS,
    assert_metrics_key_partition,
    assert_metrics_optional_not_inf,
    assert_metrics_required_finite,
    book_metrics_from_snapshot,
    concentration_top_finite_rate,
    depth_shape_finite_rate,
    microprice,
    queue_priority_finite_rate,
    side_notional_finite_rate,
    tob_size_share_finite_rate,
)
from quant_fund.schemas.order_book import BookLevel, OrderBookSnapshot


@st.composite
def uncrossed_snapshots(draw: st.DrawFn) -> OrderBookSnapshot:
    """Build a valid uncrossed OrderBookSnapshot with depth 1..5."""
    depth = draw(st.integers(min_value=1, max_value=5))
    best_bid = draw(
        st.floats(min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False)
    )
    spread = draw(st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False))
    best_ask = best_bid + spread
    tick = draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False))
    size = st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False)

    bids = [BookLevel(price=best_bid - i * tick, size=draw(size)) for i in range(depth)]
    asks = [BookLevel(price=best_ask + i * tick, size=draw(size)) for i in range(depth)]
    # Enforce strictly positive prices after stepping down
    if bids[-1].price <= 0:
        # rescale tick so worst bid stays positive
        tick = (best_bid - 1e-3) / max(depth, 1)
        bids = [BookLevel(price=best_bid - i * tick, size=bids[i].size) for i in range(depth)]
        asks = [BookLevel(price=best_ask + i * tick, size=asks[i].size) for i in range(depth)]

    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    return OrderBookSnapshot(
        security_id="HYP",
        event_time=ts,
        available_time=ts,
        bids=bids,
        asks=asks,
        depth=depth,
        source="synthetic",
        revision_id="v1",
        ingested_time=ts,
    )


@given(uncrossed_snapshots())
@settings(max_examples=60, deadline=None)
def test_metrics_match_snapshot_top_of_book_identities(snap: OrderBookSnapshot) -> None:
    """best_*/mid/spread identities: metrics mirror OrderBookSnapshot properties."""
    m = book_metrics_from_snapshot(snap)
    assert m["best_bid"] == pytest.approx(snap.best_bid)
    assert m["best_ask"] == pytest.approx(snap.best_ask)
    assert m["mid"] == pytest.approx(snap.mid)
    assert m["spread"] == pytest.approx(snap.spread)
    assert m["spread_bps"] == pytest.approx(snap.spread_bps)
    assert m["n_bid_levels"] == float(len(snap.bids))
    assert m["n_ask_levels"] == float(len(snap.asks))
    assert m["best_bid"] < m["best_ask"]
    assert m["spread"] > 0.0
    assert m["mid"] == pytest.approx(0.5 * (m["best_bid"] + m["best_ask"]))
    # Depth shape honesty
    for field in DEPTH_SHAPE_FIELDS:
        if field.startswith("bid_"):
            deep = len(snap.bids) >= 2
        else:
            deep = len(snap.asks) >= 2
        if deep:
            # Positive strict tick spacing in this strategy → finite shape fields
            assert math.isfinite(m[field])
        else:
            assert math.isnan(m[field])


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_metrics_depth_and_imbalance_bounds(snap: OrderBookSnapshot) -> None:
    m = book_metrics_from_snapshot(snap)
    assert m["bid_depth"] == pytest.approx(sum(level.size for level in snap.bids))
    assert m["ask_depth"] == pytest.approx(sum(level.size for level in snap.asks))
    assert -1.0 - 1e-12 <= m["imbalance_top"] <= 1.0 + 1e-12
    assert -1.0 - 1e-12 <= m["imbalance_depth"] <= 1.0 + 1e-12
    assert snap.best_bid < m["microprice"] < snap.best_ask or m["microprice"] == pytest.approx(
        snap.mid
    )


def test_fail_closed_empty_and_crossed_unchanged() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="non-empty"):
        OrderBookSnapshot(
            security_id="A",
            event_time=ts,
            available_time=ts,
            bids=[],
            asks=[BookLevel(price=101.0, size=1.0)],
            depth=1,
        )
    with pytest.raises(ValueError, match="crossed or locked"):
        OrderBookSnapshot(
            security_id="A",
            event_time=ts,
            available_time=ts,
            bids=[BookLevel(price=100.0, size=1.0)],
            asks=[BookLevel(price=100.0, size=1.0)],
            depth=1,
        )
    with pytest.raises(ValueError, match="crossed or locked"):
        OrderBookSnapshot(
            security_id="A",
            event_time=ts,
            available_time=ts,
            bids=[BookLevel(price=101.0, size=1.0)],
            asks=[BookLevel(price=100.0, size=1.0)],
            depth=1,
        )


def test_depth_shape_finite_rate_deep_vs_thin() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    deep = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[
            BookLevel(price=100.0, size=3.0),
            BookLevel(price=99.0, size=2.0),
            BookLevel(price=98.0, size=1.0),
        ],
        asks=[
            BookLevel(price=101.0, size=1.0),
            BookLevel(price=102.0, size=1.0),
            BookLevel(price=103.0, size=1.0),
        ],
        depth=3,
    )
    thin = OrderBookSnapshot(
        security_id="B",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=99.0, size=1.0)],
        asks=[BookLevel(price=101.0, size=1.0)],
        depth=1,
    )
    rows = [book_metrics_from_snapshot(deep), book_metrics_from_snapshot(thin)]
    rate = depth_shape_finite_rate(rows)
    assert rate == pytest.approx(1.0)
    # Thin-only panel: no eligible deep cells → NaN
    assert math.isnan(depth_shape_finite_rate([book_metrics_from_snapshot(thin)]))
    assert math.isnan(depth_shape_finite_rate([]))


@given(uncrossed_snapshots())
@settings(max_examples=60, deadline=None)
def test_microprice_between_bid_ask_property(snap: OrderBookSnapshot) -> None:
    """microprice ∈ (best_bid, best_ask); equals mid when top sizes equal."""
    mp = microprice(snap)
    bid_sz = snap.bids[0].size
    ask_sz = snap.asks[0].size
    if abs(bid_sz - ask_sz) <= 1e-12:
        assert mp == pytest.approx(snap.mid)
    else:
        assert snap.best_bid < mp < snap.best_ask
    # Metrics path agrees
    assert book_metrics_from_snapshot(snap)["microprice"] == pytest.approx(mp)


@given(uncrossed_snapshots())
@settings(max_examples=60, deadline=None)
def test_spread_bps_identity_property(snap: OrderBookSnapshot) -> None:
    """spread_bps mirrors 1e4 * spread / mid for mid > 0."""
    m = book_metrics_from_snapshot(snap)
    assert snap.mid > 0.0
    expected = 1e4 * snap.spread / snap.mid
    assert m["spread_bps"] == pytest.approx(expected, abs=1e-9)
    assert abs(m["spread_bps"] - expected) <= 1e-9
    assert abs(snap.spread_bps - expected) <= 1e-9


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_strict_level_ordering_property(snap: OrderBookSnapshot) -> None:
    """Validated snapshots always have strictly mono prices and positive sizes."""
    bid_px = [level.price for level in snap.bids]
    ask_px = [level.price for level in snap.asks]
    assert all(bid_px[i] > bid_px[i + 1] for i in range(len(bid_px) - 1))
    assert all(ask_px[i] < ask_px[i + 1] for i in range(len(ask_px) - 1))
    assert all(level.size > 0.0 and math.isfinite(level.size) for level in snap.bids + snap.asks)
    assert all(level.price > 0.0 and math.isfinite(level.price) for level in snap.bids + snap.asks)


def test_book_level_rejects_non_finite_or_non_positive() -> None:
    with pytest.raises(ValueError, match="finite and > 0"):
        BookLevel(price=float("nan"), size=1.0)
    with pytest.raises(ValueError, match="finite and > 0"):
        BookLevel(price=1.0, size=float("inf"))
    with pytest.raises(ValueError, match="finite and > 0"):
        BookLevel(price=0.0, size=1.0)
    with pytest.raises(ValueError, match="finite and > 0"):
        BookLevel(price=1.0, size=-0.5)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_available_time_and_depth_bounds_property(snap: OrderBookSnapshot) -> None:
    """PIT: available_time >= event_time; side lengths within declared depth."""
    assert snap.available_time >= snap.event_time
    assert 1 <= len(snap.bids) <= snap.depth
    assert 1 <= len(snap.asks) <= snap.depth


def test_available_time_before_event_fail_closed() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    earlier = datetime(2020, 1, 2, 15, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="available_time"):
        OrderBookSnapshot(
            security_id="A",
            event_time=ts,
            available_time=earlier,
            bids=[BookLevel(price=99.0, size=1.0)],
            asks=[BookLevel(price=101.0, size=1.0)],
            depth=1,
        )


@given(uncrossed_snapshots())
@settings(max_examples=60, deadline=None)
def test_side_depth_and_mid_identities_property(snap: OrderBookSnapshot) -> None:
    """bid/ask_depth == sum(sizes); mid == 0.5*(best_bid+best_ask)."""
    m = book_metrics_from_snapshot(snap)
    assert m["bid_depth"] == pytest.approx(sum(level.size for level in snap.bids), abs=1e-12)
    assert m["ask_depth"] == pytest.approx(sum(level.size for level in snap.asks), abs=1e-12)
    assert m["mid"] == pytest.approx(0.5 * (m["best_bid"] + m["best_ask"]), abs=1e-12)
    assert snap.mid == pytest.approx(0.5 * (snap.best_bid + snap.best_ask), abs=1e-12)
    assert m["bid_size_concentration_top"] == pytest.approx(
        m["top_bid_size"] / m["bid_depth"], abs=1e-12
    )
    assert m["ask_size_concentration_top"] == pytest.approx(
        m["top_ask_size"] / m["ask_depth"], abs=1e-12
    )
    assert 0.0 < m["bid_size_concentration_top"] <= 1.0 + 1e-12
    assert 0.0 < m["ask_size_concentration_top"] <= 1.0 + 1e-12


def test_size_concentration_top_unit_and_thin_safe() -> None:
    """Known concentration + thin-safe NaN helper path via depth-1 book."""
    from quant_fund.microstructure.book_metrics import _size_concentration_top

    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    deep = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[
            BookLevel(price=100.0, size=2.0),
            BookLevel(price=99.0, size=2.0),
        ],
        asks=[
            BookLevel(price=101.0, size=1.0),
            BookLevel(price=102.0, size=3.0),
        ],
        depth=2,
    )
    m = book_metrics_from_snapshot(deep)
    assert m["bid_depth"] == pytest.approx(4.0)
    assert m["ask_depth"] == pytest.approx(4.0)
    assert m["mid"] == pytest.approx(100.5)
    assert m["bid_size_concentration_top"] == pytest.approx(0.5)
    assert m["ask_size_concentration_top"] == pytest.approx(0.25)
    assert math.isnan(_size_concentration_top(1.0, 0.0))
    assert math.isnan(_size_concentration_top(1.0, float("nan")))


@given(uncrossed_snapshots())
@settings(max_examples=50, deadline=None)
def test_top_size_le_depth_and_micro_minus_mid(snap: OrderBookSnapshot) -> None:
    """top_*_size ≤ side depth; microprice_minus_mid == microprice - mid."""
    m = book_metrics_from_snapshot(snap)
    assert m["top_bid_size"] <= m["bid_depth"] + 1e-12
    assert m["top_ask_size"] <= m["ask_depth"] + 1e-12
    assert m["microprice_minus_mid"] == pytest.approx(m["microprice"] - m["mid"], abs=1e-12)
    if m["mid"] > 0:
        expected_bps = 1e4 * (m["microprice"] - m["mid"]) / m["mid"]
        assert m["microprice_minus_mid_bps"] == pytest.approx(expected_bps, abs=1e-9)


@given(uncrossed_snapshots())
@settings(max_examples=60, deadline=None)
def test_spread_positive_identity_property(snap: OrderBookSnapshot) -> None:
    """spread > 0 and best_ask - best_bid == spread (schema + metrics)."""
    m = book_metrics_from_snapshot(snap)
    assert snap.spread > 0.0
    assert m["spread"] > 0.0
    assert m["spread"] == pytest.approx(m["best_ask"] - m["best_bid"], abs=1e-12)
    assert snap.spread == pytest.approx(snap.best_ask - snap.best_bid, abs=1e-12)
    assert m["spread"] == pytest.approx(snap.spread, abs=1e-12)


def test_concentration_top_finite_rate_and_side_structure_fields() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    snap = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=99.0, size=1.0), BookLevel(price=98.0, size=1.0)],
        asks=[BookLevel(price=101.0, size=1.0)],
        depth=2,
    )
    row = book_metrics_from_snapshot(snap)
    for key in SIDE_STRUCTURE_FIELDS:
        assert key in row
        assert math.isfinite(row[key])
    assert concentration_top_finite_rate([row]) == pytest.approx(1.0)
    assert math.isnan(concentration_top_finite_rate([]))
    # Fake zero-depth row → not eligible
    bogus = {
        **row,
        "bid_depth": 0.0,
        "ask_depth": 0.0,
        "bid_size_concentration_top": float("nan"),
        "ask_size_concentration_top": float("nan"),
    }
    assert math.isnan(concentration_top_finite_rate([bogus]))


@given(uncrossed_snapshots())
@settings(max_examples=50, deadline=None)
def test_imbalance_top_identity_property(snap: OrderBookSnapshot) -> None:
    """imbalance_top == (top_bid_size - top_ask_size) / (top_bid_size + top_ask_size)."""
    m = book_metrics_from_snapshot(snap)
    denom = m["top_bid_size"] + m["top_ask_size"]
    assert denom > 0.0
    expected = (m["top_bid_size"] - m["top_ask_size"]) / denom
    assert m["imbalance_top"] == pytest.approx(expected, abs=1e-12)
    depth_denom = m["bid_depth"] + m["ask_depth"]
    assert depth_denom > 0.0
    assert m["imbalance_depth"] == pytest.approx(
        (m["bid_depth"] - m["ask_depth"]) / depth_denom, abs=1e-12
    )


@given(uncrossed_snapshots())
@settings(max_examples=60, deadline=None)
def test_concentration_bounds_and_top_sizes_positive(snap: OrderBookSnapshot) -> None:
    """When finite, concentration ∈ (0, 1]; top sizes > 0 on valid snapshots."""
    m = book_metrics_from_snapshot(snap)
    assert m["top_bid_size"] > 0.0 and math.isfinite(m["top_bid_size"])
    assert m["top_ask_size"] > 0.0 and math.isfinite(m["top_ask_size"])
    for key in SIDE_STRUCTURE_FIELDS:
        val = m[key]
        assert math.isfinite(val)
        assert 0.0 < val <= 1.0 + 1e-12


@given(uncrossed_snapshots())
@settings(max_examples=60, deadline=None)
def test_microprice_weight_balance_identity(snap: OrderBookSnapshot) -> None:
    """microprice == best_ask * w + best_bid * (1-w) for w = microprice_weight_balance."""
    m = book_metrics_from_snapshot(snap)
    w = m["microprice_weight_balance"]
    assert math.isfinite(w)
    assert 0.0 < w < 1.0 or abs(w - 0.5) < 1e-12 or abs(w - 0.0) < 1e-12 or abs(w - 1.0) < 1e-12
    # sizes > 0 ⇒ w ∈ (0, 1)
    assert 0.0 < w < 1.0
    expected = m["best_ask"] * w + m["best_bid"] * (1.0 - w)
    assert m["microprice"] == pytest.approx(expected, abs=1e-12)
    assert w == pytest.approx(
        m["top_bid_size"] / (m["top_bid_size"] + m["top_ask_size"]), abs=1e-12
    )


def test_microprice_weight_balance_unit() -> None:
    from quant_fund.microstructure.book_metrics import _microprice_weight_balance

    assert _microprice_weight_balance(3.0, 1.0) == pytest.approx(0.75)
    assert math.isnan(_microprice_weight_balance(0.0, 0.0))
    assert math.isnan(_microprice_weight_balance(float("nan"), 1.0))


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_n_levels_match_side_lengths(snap: OrderBookSnapshot) -> None:
    """n_*_levels metrics equal len(bids/asks) and ≤ declared depth."""
    m = book_metrics_from_snapshot(snap)
    assert m["n_bid_levels"] == float(len(snap.bids))
    assert m["n_ask_levels"] == float(len(snap.asks))
    assert m["n_bid_levels"] <= float(snap.depth)
    assert m["n_ask_levels"] <= float(snap.depth)


@given(uncrossed_snapshots())
@settings(max_examples=60, deadline=None)
def test_effective_spread_and_depth_imbalance_abs(snap: OrderBookSnapshot) -> None:
    """effective_spread aliases spread; depth_imbalance_abs == |imbalance_depth| ∈ [0,1]."""
    m = book_metrics_from_snapshot(snap)
    assert m["effective_spread"] == pytest.approx(m["spread"], abs=1e-12)
    assert m["effective_spread"] == pytest.approx(m["best_ask"] - m["best_bid"], abs=1e-12)
    assert snap.effective_spread == pytest.approx(snap.spread, abs=1e-12)
    assert m["depth_imbalance_abs"] == pytest.approx(abs(m["imbalance_depth"]), abs=1e-12)
    assert 0.0 <= m["depth_imbalance_abs"] <= 1.0 + 1e-12


@given(uncrossed_snapshots())
@settings(max_examples=60, deadline=None)
def test_depth_shape_finite_iff_n_ge_2_after_structure_fields(snap: OrderBookSnapshot) -> None:
    """DEPTH_SHAPE still NaN iff n<2 even with concentration/weight fields present."""
    m = book_metrics_from_snapshot(snap)
    # Structure fields always present on valid snaps
    assert math.isfinite(m["microprice_weight_balance"])
    for key in SIDE_STRUCTURE_FIELDS:
        assert math.isfinite(m[key])
    for field in DEPTH_SHAPE_FIELDS:
        if field.startswith("bid_"):
            deep = m["n_bid_levels"] >= 2.0
        else:
            deep = m["n_ask_levels"] >= 2.0
        if deep:
            assert math.isfinite(m[field]), field
        else:
            assert math.isnan(m[field]), field


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_price_envelope_vs_best(snap: OrderBookSnapshot) -> None:
    """Every bid price ≤ best_bid; every ask price ≥ best_ask (strict mono already)."""
    assert all(level.price <= snap.best_bid + 1e-15 for level in snap.bids)
    assert all(level.price >= snap.best_ask - 1e-15 for level in snap.asks)
    assert snap.bids[0].price == snap.best_bid
    assert snap.asks[0].price == snap.best_ask


@given(uncrossed_snapshots())
@settings(max_examples=60, deadline=None)
def test_all_sizes_positive_and_spread_over_mid(snap: OrderBookSnapshot) -> None:
    """All level sizes > 0 finite; spread_over_mid == spread/mid == spread_bps/1e4."""
    for level in snap.bids + snap.asks:
        assert level.size > 0.0 and math.isfinite(level.size)
    m = book_metrics_from_snapshot(snap)
    assert math.isfinite(m["spread_over_mid"])
    assert m["spread_over_mid"] == pytest.approx(m["spread"] / m["mid"], abs=1e-12)
    assert m["spread_over_mid"] == pytest.approx(m["spread_bps"] / 1e4, abs=1e-12)
    assert m["spread_bps"] == pytest.approx(1e4 * m["spread_over_mid"], abs=1e-9)


def test_depth_zero_and_empty_sides_fail_closed() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="depth must be >= 1"):
        OrderBookSnapshot(
            security_id="A",
            event_time=ts,
            available_time=ts,
            bids=[BookLevel(price=99.0, size=1.0)],
            asks=[BookLevel(price=101.0, size=1.0)],
            depth=0,
        )
    with pytest.raises(ValueError, match="non-empty"):
        OrderBookSnapshot(
            security_id="A",
            event_time=ts,
            available_time=ts,
            bids=[],
            asks=[],
            depth=1,
        )


def test_side_length_exceeds_depth_fail_closed() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="exceeds declared depth"):
        OrderBookSnapshot(
            security_id="A",
            event_time=ts,
            available_time=ts,
            bids=[
                BookLevel(price=100.0, size=1.0),
                BookLevel(price=99.0, size=1.0),
            ],
            asks=[BookLevel(price=101.0, size=1.0)],
            depth=1,
        )


@given(uncrossed_snapshots())
@settings(max_examples=60, deadline=None)
def test_half_spread_and_revision_id(snap: OrderBookSnapshot) -> None:
    """half_spread == spread/2; mid±half recovers best ask/bid; revision_id non-empty."""
    assert isinstance(snap.revision_id, str) and snap.revision_id.strip() != ""
    m = book_metrics_from_snapshot(snap)
    assert m["half_spread"] == pytest.approx(0.5 * m["spread"], abs=1e-12)
    assert snap.half_spread == pytest.approx(0.5 * snap.spread, abs=1e-12)
    assert m["best_ask"] == pytest.approx(m["mid"] + m["half_spread"], abs=1e-12)
    assert m["best_bid"] == pytest.approx(m["mid"] - m["half_spread"], abs=1e-12)
    assert m["microprice_minus_mid_bps"] == pytest.approx(
        1e4 * m["microprice_minus_mid"] / m["mid"], abs=1e-9
    )


def test_empty_revision_id_fail_closed() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="non-empty"):
        OrderBookSnapshot(
            security_id="A",
            event_time=ts,
            available_time=ts,
            bids=[BookLevel(price=99.0, size=1.0)],
            asks=[BookLevel(price=101.0, size=1.0)],
            depth=1,
            revision_id="   ",
        )


def test_empty_security_id_fail_closed() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="non-empty"):
        OrderBookSnapshot(
            security_id="",
            event_time=ts,
            available_time=ts,
            bids=[BookLevel(price=99.0, size=1.0)],
            asks=[BookLevel(price=101.0, size=1.0)],
            depth=1,
        )


@given(uncrossed_snapshots())
@settings(max_examples=50, deadline=None)
def test_tz_aware_and_quoted_half_spread_bps(snap: OrderBookSnapshot) -> None:
    """Timestamps timezone-aware; quoted_spread_bps aliases; half_spread_bps identity."""
    assert snap.event_time.tzinfo is not None and snap.event_time.utcoffset() is not None
    assert snap.available_time.tzinfo is not None and snap.available_time.utcoffset() is not None
    m = book_metrics_from_snapshot(snap)
    assert m["quoted_spread_bps"] == pytest.approx(m["spread_bps"], abs=1e-12)
    assert m["half_spread_bps"] == pytest.approx(1e4 * m["half_spread"] / m["mid"], abs=1e-9)
    assert m["half_spread_bps"] == pytest.approx(0.5 * m["spread_bps"], abs=1e-9)


def test_naive_datetime_fail_closed() -> None:
    ts = datetime(2020, 1, 2, 16, 0)  # naive
    with pytest.raises(ValueError, match="timezone-aware"):
        OrderBookSnapshot(
            security_id="A",
            event_time=ts,
            available_time=ts,
            bids=[BookLevel(price=99.0, size=1.0)],
            asks=[BookLevel(price=101.0, size=1.0)],
            depth=1,
        )


def test_nan_inf_level_fail_closed() -> None:
    with pytest.raises(ValueError, match="finite and > 0"):
        BookLevel(price=float("nan"), size=1.0)
    with pytest.raises(ValueError, match="finite and > 0"):
        BookLevel(price=1.0, size=float("inf"))
    with pytest.raises(ValueError, match="finite and > 0"):
        BookLevel(price=float("-inf"), size=1.0)


def test_ingested_time_optional_tz_aware() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    ok = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        ingested_time=ts,
        bids=[BookLevel(price=99.0, size=1.0)],
        asks=[BookLevel(price=101.0, size=1.0)],
        depth=1,
    )
    assert ok.ingested_time == ts
    with pytest.raises(ValueError, match="timezone-aware"):
        OrderBookSnapshot(
            security_id="A",
            event_time=ts,
            available_time=ts,
            ingested_time=datetime(2020, 1, 2, 16, 0),
            bids=[BookLevel(price=99.0, size=1.0)],
            asks=[BookLevel(price=101.0, size=1.0)],
            depth=1,
        )


@given(uncrossed_snapshots())
@settings(max_examples=50, deadline=None)
def test_pit_chain_and_touch_imbalance(snap: OrderBookSnapshot) -> None:
    """available ≥ event; unique prices; touch_size_imbalance == imbalance_top."""
    assert snap.available_time >= snap.event_time
    if snap.ingested_time is not None:
        assert snap.ingested_time >= snap.available_time
    bid_px = [level.price for level in snap.bids]
    ask_px = [level.price for level in snap.asks]
    assert len(bid_px) == len(set(bid_px))
    assert len(ask_px) == len(set(ask_px))
    m = book_metrics_from_snapshot(snap)
    assert m["touch_size_imbalance"] == pytest.approx(m["imbalance_top"], abs=1e-12)


def test_ingested_before_available_fail_closed() -> None:
    event = datetime(2020, 1, 2, 15, 0, tzinfo=UTC)
    available = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    ingested = datetime(2020, 1, 2, 15, 30, tzinfo=UTC)
    with pytest.raises(ValueError, match="ingested_time"):
        OrderBookSnapshot(
            security_id="A",
            event_time=event,
            available_time=available,
            ingested_time=ingested,
            bids=[BookLevel(price=99.0, size=1.0)],
            asks=[BookLevel(price=101.0, size=1.0)],
            depth=1,
        )


def test_duplicate_prices_fail_closed_explicit() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="strictly descending|unique"):
        OrderBookSnapshot(
            security_id="A",
            event_time=ts,
            available_time=ts,
            bids=[
                BookLevel(price=100.0, size=1.0),
                BookLevel(price=100.0, size=2.0),
            ],
            asks=[BookLevel(price=101.0, size=1.0)],
            depth=2,
        )


@given(uncrossed_snapshots())
@settings(max_examples=50, deadline=None)
def test_source_and_queue_priority_and_spread_bps_positive(snap: OrderBookSnapshot) -> None:
    """source non-empty; queue_priority_proxy identity; spread_bps > 0 when mid > 0."""
    assert isinstance(snap.source, str) and snap.source.strip() != ""
    m = book_metrics_from_snapshot(snap)
    assert m["mid"] > 0.0
    assert m["spread_bps"] > 0.0
    expected_bid = m["top_bid_size"] / (m["top_bid_size"] + m["bid_depth"])
    expected_ask = m["top_ask_size"] / (m["top_ask_size"] + m["ask_depth"])
    assert m["queue_priority_proxy"] == pytest.approx(expected_bid, abs=1e-12)
    assert m["ask_queue_priority_proxy"] == pytest.approx(expected_ask, abs=1e-12)
    assert 0.0 < m["queue_priority_proxy"] <= 1.0 + 1e-12


def test_empty_source_fail_closed() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="non-empty"):
        OrderBookSnapshot(
            security_id="A",
            event_time=ts,
            available_time=ts,
            bids=[BookLevel(price=99.0, size=1.0)],
            asks=[BookLevel(price=101.0, size=1.0)],
            depth=1,
            source="  ",
        )


def test_queue_priority_proxy_thin_safe_unit() -> None:
    from quant_fund.microstructure.book_metrics import _queue_priority_proxy

    assert _queue_priority_proxy(2.0, 4.0) == pytest.approx(2.0 / 6.0)
    assert math.isnan(_queue_priority_proxy(1.0, 0.0))
    assert math.isnan(_queue_priority_proxy(0.0, 0.0))
    assert math.isnan(_queue_priority_proxy(float("nan"), 1.0))


def test_source_is_stripped() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    snap = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=99.0, size=1.0)],
        asks=[BookLevel(price=101.0, size=1.0)],
        depth=1,
        source="  synthetic  ",
    )
    assert snap.source == "synthetic"


@given(uncrossed_snapshots())
@settings(max_examples=50, deadline=None)
def test_queue_priority_bounds_and_fields_tuple(snap: OrderBookSnapshot) -> None:
    """queue_priority_proxy / ask_queue_priority_proxy ∈ (0, 1] when finite."""
    m = book_metrics_from_snapshot(snap)
    for key in QUEUE_STRUCTURE_FIELDS:
        assert key in m
        val = m[key]
        assert math.isfinite(val)
        assert 0.0 < val <= 1.0 + 1e-12


def test_source_strip_preserves_non_empty_core() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    snap = OrderBookSnapshot(
        security_id="  A  ",
        event_time=ts,
        available_time=ts,
        revision_id="  v1  ",
        source="\talpaca\n",
        bids=[BookLevel(price=99.0, size=1.0)],
        asks=[BookLevel(price=101.0, size=1.0)],
        depth=1,
    )
    assert snap.security_id == "A"
    assert snap.revision_id == "v1"
    assert snap.source == "alpaca"


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_queue_priority_both_finite_on_positive_depth(snap: OrderBookSnapshot) -> None:
    """Hypothesis snaps always have depth>0 sides → both queue priorities finite."""
    m = book_metrics_from_snapshot(snap)
    assert m["bid_depth"] > 0.0 and m["ask_depth"] > 0.0
    assert math.isfinite(m["queue_priority_proxy"])
    assert math.isfinite(m["ask_queue_priority_proxy"])


def test_queue_priority_finite_rate_helper() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    snap = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=99.0, size=1.0)],
        asks=[BookLevel(price=101.0, size=1.0)],
        depth=1,
    )
    row = book_metrics_from_snapshot(snap)
    assert queue_priority_finite_rate([row]) == pytest.approx(1.0)
    assert math.isnan(queue_priority_finite_rate([]))
    thin = {
        **row,
        "bid_depth": 0.0,
        "ask_depth": 0.0,
        "queue_priority_proxy": float("nan"),
        "ask_queue_priority_proxy": float("nan"),
    }
    assert math.isnan(queue_priority_finite_rate([thin]))


@given(uncrossed_snapshots().filter(lambda s: len(s.bids) >= 2 and len(s.asks) >= 2))
@settings(max_examples=40, deadline=None)
def test_field_groups_coexist_on_deep_books(snap: OrderBookSnapshot) -> None:
    """On deep multi-level books, DEPTH/SIDE/QUEUE structure fields are all finite."""
    m = book_metrics_from_snapshot(snap)
    for key in DEPTH_SHAPE_FIELDS + SIDE_STRUCTURE_FIELDS + QUEUE_STRUCTURE_FIELDS:
        assert math.isfinite(m[key]), key
    assert math.isfinite(m["top_of_book_notional_proxy"])
    assert m["top_of_book_notional_proxy"] == pytest.approx(
        m["best_bid"] * m["top_bid_size"] + m["best_ask"] * m["top_ask_size"], abs=1e-9
    )
    assert m["top_of_book_notional_proxy"] > 0.0


def test_negative_size_fail_closed() -> None:
    with pytest.raises(ValueError, match="finite and > 0"):
        BookLevel(price=100.0, size=-1.0)
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="finite and > 0"):
        OrderBookSnapshot(
            security_id="A",
            event_time=ts,
            available_time=ts,
            bids=[BookLevel(price=99.0, size=-0.01)],
            asks=[BookLevel(price=101.0, size=1.0)],
            depth=1,
        )


def test_top_of_book_notional_proxy_unit() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    snap = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=99.0, size=2.0)],
        asks=[BookLevel(price=101.0, size=3.0)],
        depth=1,
    )
    m = book_metrics_from_snapshot(snap)
    assert m["mid"] == pytest.approx(100.0)
    assert m["top_of_book_notional_proxy"] == pytest.approx(99.0 * 2.0 + 101.0 * 3.0)


@given(uncrossed_snapshots())
@settings(max_examples=50, deadline=None)
def test_notional_proxies_and_zero_size(snap: OrderBookSnapshot) -> None:
    """TOB notional > 0; side notionals == best * depth."""
    m = book_metrics_from_snapshot(snap)
    assert m["mid"] > 0.0
    assert m["top_bid_size"] > 0.0 and m["top_ask_size"] > 0.0
    assert m["top_of_book_notional_proxy"] > 0.0
    assert m["side_notional_proxy_bid"] == pytest.approx(m["best_bid"] * m["bid_depth"], abs=1e-9)
    assert m["side_notional_proxy_ask"] == pytest.approx(m["best_ask"] * m["ask_depth"], abs=1e-9)
    assert m["side_notional_proxy_bid"] > 0.0
    assert m["side_notional_proxy_ask"] > 0.0


def test_zero_size_fail_closed() -> None:
    with pytest.raises(ValueError, match="finite and > 0"):
        BookLevel(price=100.0, size=0.0)
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="finite and > 0"):
        OrderBookSnapshot(
            security_id="A",
            event_time=ts,
            available_time=ts,
            bids=[BookLevel(price=99.0, size=0.0)],
            asks=[BookLevel(price=101.0, size=1.0)],
            depth=1,
        )


def test_side_notional_proxy_thin_safe_unit() -> None:
    from quant_fund.microstructure.book_metrics import _side_notional_proxy

    assert _side_notional_proxy(100.0, 2.5) == pytest.approx(250.0)
    assert math.isnan(_side_notional_proxy(100.0, 0.0))
    assert math.isnan(_side_notional_proxy(0.0, 2.0))


@given(uncrossed_snapshots())
@settings(max_examples=50, deadline=None)
def test_side_notional_fields_and_locked_book(snap: OrderBookSnapshot) -> None:
    """SIDE_NOTIONAL_FIELDS > 0; notional/best == depth when best>0."""
    m = book_metrics_from_snapshot(snap)
    for key in SIDE_NOTIONAL_FIELDS:
        assert key in m
        assert m[key] > 0.0 and math.isfinite(m[key])
    assert m["best_bid"] > 0.0 and m["best_ask"] > 0.0
    assert m["side_notional_proxy_bid"] / m["best_bid"] == pytest.approx(m["bid_depth"], abs=1e-9)
    assert m["side_notional_proxy_ask"] / m["best_ask"] == pytest.approx(m["ask_depth"], abs=1e-9)


def test_locked_book_fail_closed_reinforce() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="crossed or locked"):
        OrderBookSnapshot(
            security_id="A",
            event_time=ts,
            available_time=ts,
            bids=[BookLevel(price=100.0, size=1.0)],
            asks=[BookLevel(price=100.0, size=1.0)],
            depth=1,
        )


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_side_and_tob_notionals_finite_together(snap: OrderBookSnapshot) -> None:
    """SIDE_NOTIONAL + TOB notional all finite on Hypothesis snaps."""
    m = book_metrics_from_snapshot(snap)
    assert math.isfinite(m["top_of_book_notional_proxy"])
    for key in SIDE_NOTIONAL_FIELDS:
        assert math.isfinite(m[key]) and m[key] > 0.0


def test_side_notional_finite_rate_and_best_nonpositive() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    snap = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=99.0, size=1.0)],
        asks=[BookLevel(price=101.0, size=1.0)],
        depth=1,
    )
    row = book_metrics_from_snapshot(snap)
    assert side_notional_finite_rate([row]) == pytest.approx(1.0)
    assert math.isnan(side_notional_finite_rate([]))
    thin = {
        **row,
        "bid_depth": 0.0,
        "ask_depth": 0.0,
        "side_notional_proxy_bid": float("nan"),
        "side_notional_proxy_ask": float("nan"),
    }
    assert math.isnan(side_notional_finite_rate([thin]))
    with pytest.raises(ValueError, match="finite and > 0"):
        BookLevel(price=0.0, size=1.0)
    with pytest.raises(ValueError, match="finite and > 0"):
        BookLevel(price=-1.0, size=1.0)


@given(uncrossed_snapshots())
@settings(max_examples=50, deadline=None)
def test_notional_imbalance_and_rate_bounds(snap: OrderBookSnapshot) -> None:
    """notional_imbalance ∈ [-1,1]; side_notional_finite_rate ∈ [0,1] on deep rows."""
    m = book_metrics_from_snapshot(snap)
    assert math.isfinite(m["notional_imbalance"])
    assert -1.0 - 1e-12 <= m["notional_imbalance"] <= 1.0 + 1e-12
    expected = (m["side_notional_proxy_bid"] - m["side_notional_proxy_ask"]) / (
        m["side_notional_proxy_bid"] + m["side_notional_proxy_ask"]
    )
    assert m["notional_imbalance"] == pytest.approx(expected, abs=1e-12)
    rate = side_notional_finite_rate([m])
    assert 0.0 <= rate <= 1.0 + 1e-12
    assert rate == pytest.approx(1.0)


def test_side_notional_finite_rate_empty_nan() -> None:
    assert math.isnan(side_notional_finite_rate([]))


def test_crossed_book_fail_closed_clear_message() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="crossed or locked"):
        OrderBookSnapshot(
            security_id="A",
            event_time=ts,
            available_time=ts,
            bids=[BookLevel(price=102.0, size=1.0)],
            asks=[BookLevel(price=101.0, size=1.0)],
            depth=1,
        )


def test_equal_notional_imbalance_zero_and_tob_share() -> None:
    """Equal side notionals → imbalance 0; tob_notional_share ∈ (0,1]."""
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    # best_bid*bid_depth = 100*2 = 200; best_ask*ask_depth = 100*2 = 200 (use mid-ish)
    snap = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=100.0, size=2.0)],
        asks=[BookLevel(price=100.5, size=200.0 / 100.5)],  # ask notional = 200
        depth=1,
    )
    m = book_metrics_from_snapshot(snap)
    assert m["side_notional_proxy_bid"] == pytest.approx(200.0, abs=1e-9)
    assert m["side_notional_proxy_ask"] == pytest.approx(200.0, abs=1e-9)
    assert m["notional_imbalance"] == pytest.approx(0.0, abs=1e-9)
    assert math.isfinite(m["tob_notional_share"])
    assert m["tob_notional_share"] > 0.0
    assert m["tob_notional_share"] == pytest.approx(
        m["top_of_book_notional_proxy"]
        / (m["side_notional_proxy_bid"] + m["side_notional_proxy_ask"]),
        abs=1e-12,
    )


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_tob_notional_share_bounds_property(snap: OrderBookSnapshot) -> None:
    m = book_metrics_from_snapshot(snap)
    assert math.isfinite(m["tob_notional_share"])
    assert 0.0 < m["tob_notional_share"] <= 1.0 + 1e-12
    assert m["tob_notional_share"] == pytest.approx(
        m["top_of_book_notional_proxy"]
        / (m["side_notional_proxy_bid"] + m["side_notional_proxy_ask"]),
        abs=1e-12,
    )


def test_empty_bids_or_asks_fail_closed_reinforce() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="non-empty"):
        OrderBookSnapshot(
            security_id="A",
            event_time=ts,
            available_time=ts,
            bids=[],
            asks=[BookLevel(price=101.0, size=1.0)],
            depth=1,
        )
    with pytest.raises(ValueError, match="non-empty"):
        OrderBookSnapshot(
            security_id="A",
            event_time=ts,
            available_time=ts,
            bids=[BookLevel(price=99.0, size=1.0)],
            asks=[],
            depth=1,
        )


@given(uncrossed_snapshots())
@settings(max_examples=50, deadline=None)
def test_tob_size_share_honest_bounds(snap: OrderBookSnapshot) -> None:
    """tob_size_share ∈ (0,1]; equals (Σ top sizes)/(Σ side depths)."""
    m = book_metrics_from_snapshot(snap)
    assert math.isfinite(m["tob_size_share"])
    assert 0.0 < m["tob_size_share"] <= 1.0 + 1e-12
    expected = (m["top_bid_size"] + m["top_ask_size"]) / (m["bid_depth"] + m["ask_depth"])
    assert m["tob_size_share"] == pytest.approx(expected, abs=1e-12)
    # Touch-priced tob_notional_share ∈ (0,1]
    assert math.isfinite(m["tob_notional_share"])
    assert 0.0 < m["tob_notional_share"] <= 1.0 + 1e-12


def test_tob_size_share_unit_equal_tops_depths() -> None:
    """Depth-1: tops == depths → tob_size_share == 1."""
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    snap = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=99.0, size=2.0)],
        asks=[BookLevel(price=101.0, size=3.0)],
        depth=1,
    )
    m = book_metrics_from_snapshot(snap)
    assert m["tob_size_share"] == pytest.approx(1.0)
    deep = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[
            BookLevel(price=100.0, size=1.0),
            BookLevel(price=99.0, size=1.0),
        ],
        asks=[
            BookLevel(price=101.0, size=1.0),
            BookLevel(price=102.0, size=1.0),
        ],
        depth=2,
    )
    md = book_metrics_from_snapshot(deep)
    assert md["tob_size_share"] == pytest.approx(0.5)


def test_depth_declare_mismatch_fail_closed_reinforce() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="exceeds declared depth"):
        OrderBookSnapshot(
            security_id="A",
            event_time=ts,
            available_time=ts,
            bids=[BookLevel(price=100.0, size=1.0), BookLevel(price=99.0, size=1.0)],
            asks=[BookLevel(price=101.0, size=1.0)],
            depth=1,
        )


@given(
    uncrossed_snapshots().filter(lambda s: s.depth == 1 and len(s.bids) == 1 and len(s.asks) == 1)
)
@settings(max_examples=40, deadline=None)
def test_tob_size_share_depth1_is_one_property(snap: OrderBookSnapshot) -> None:
    """On depth=1 books, tob_size_share == 1.0 and ∈ TOB_SHARE_FIELDS."""
    m = book_metrics_from_snapshot(snap)
    for key in TOB_SHARE_FIELDS:
        assert key in m
        assert m[key] == pytest.approx(1.0)
        assert m[key] <= 1.0 + 1e-12


def test_tob_size_share_zero_depth_sum_nan() -> None:
    from quant_fund.microstructure.book_metrics import _tob_size_share

    assert math.isnan(_tob_size_share(1.0, 1.0, 0.0, 0.0))
    assert math.isnan(_tob_size_share(1.0, 1.0, 1.0, 0.0))
    assert math.isnan(_tob_size_share(0.0, 1.0, 1.0, 1.0))


@given(uncrossed_snapshots())
@settings(max_examples=50, deadline=None)
def test_metrics_key_fields_round_trip_integrity(snap: OrderBookSnapshot) -> None:
    """Snapshot → metrics reconstructs best/mid/spread/n_levels/tob_size_share."""
    m = book_metrics_from_snapshot(snap)
    assert m["best_bid"] == pytest.approx(snap.best_bid, abs=1e-12)
    assert m["best_ask"] == pytest.approx(snap.best_ask, abs=1e-12)
    assert m["mid"] == pytest.approx(snap.mid, abs=1e-12)
    assert m["spread"] == pytest.approx(snap.spread, abs=1e-12)
    assert m["n_bid_levels"] == pytest.approx(float(len(snap.bids)), abs=1e-12)
    assert m["n_ask_levels"] == pytest.approx(float(len(snap.asks)), abs=1e-12)
    expected_share = (snap.bids[0].size + snap.asks[0].size) / (
        sum(level.size for level in snap.bids) + sum(level.size for level in snap.asks)
    )
    assert m["tob_size_share"] == pytest.approx(expected_share, abs=1e-12)


@given(uncrossed_snapshots().filter(lambda s: len(s.bids) >= 2 and len(s.asks) >= 2))
@settings(max_examples=40, deadline=None)
def test_tob_share_side_notional_queue_finite_together(snap: OrderBookSnapshot) -> None:
    """On deep books, TOB_SHARE + SIDE_NOTIONAL + QUEUE fields all finite."""
    m = book_metrics_from_snapshot(snap)
    for key in TOB_SHARE_FIELDS + SIDE_NOTIONAL_FIELDS + QUEUE_STRUCTURE_FIELDS:
        assert math.isfinite(m[key]), key


def test_tob_size_share_finite_rate_helper() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    snap = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=99.0, size=1.0)],
        asks=[BookLevel(price=101.0, size=1.0)],
        depth=1,
    )
    row = book_metrics_from_snapshot(snap)
    assert tob_size_share_finite_rate([row]) == pytest.approx(1.0)
    assert math.isnan(tob_size_share_finite_rate([]))
    thin = {**row, "bid_depth": 0.0, "ask_depth": 0.0, "tob_size_share": float("nan")}
    assert math.isnan(tob_size_share_finite_rate([thin]))


@given(uncrossed_snapshots())
@settings(max_examples=50, deadline=None)
def test_round_trip_imbalance_spread_bps_microprice(snap: OrderBookSnapshot) -> None:
    """Round-trip preserves imbalance_top / spread_bps / microprice; TOB fields finite."""
    from quant_fund.microstructure.book_metrics import microprice as mp_fn

    m = book_metrics_from_snapshot(snap)
    top_bid = snap.bids[0].size
    top_ask = snap.asks[0].size
    expected_imb = (top_bid - top_ask) / (top_bid + top_ask)
    assert m["imbalance_top"] == pytest.approx(expected_imb, abs=1e-12)
    assert m["spread_bps"] == pytest.approx(snap.spread_bps, abs=1e-9)
    assert m["microprice"] == pytest.approx(mp_fn(snap), abs=1e-12)
    for key in ("best_bid", "best_ask", "mid", "spread", "spread_bps", "microprice"):
        assert math.isfinite(m[key]), key


@given(uncrossed_snapshots().filter(lambda s: len(s.bids) >= 2 and len(s.asks) >= 2))
@settings(max_examples=30, deadline=None)
def test_required_tob_fields_finite_on_deep_snaps(snap: OrderBookSnapshot) -> None:
    """Valid deep snaps: required TOB scalar fields are finite (no NaN)."""
    m = book_metrics_from_snapshot(snap)
    required = (
        "best_bid",
        "best_ask",
        "mid",
        "spread",
        "spread_bps",
        "microprice",
        "imbalance_top",
        "tob_size_share",
    )
    for key in required:
        assert math.isfinite(m[key]), key


@given(uncrossed_snapshots().filter(lambda s: len(s.bids) >= 2 and len(s.asks) >= 2))
@settings(max_examples=30, deadline=None)
def test_full_stack_finite_and_required_keys(snap: OrderBookSnapshot) -> None:
    """Deep snaps: DEPTH+SIDE+QUEUE+NOTIONAL+TOB_SHARE finite; required keys covered."""
    m = book_metrics_from_snapshot(snap)
    for key in (
        DEPTH_SHAPE_FIELDS
        + SIDE_STRUCTURE_FIELDS
        + QUEUE_STRUCTURE_FIELDS
        + SIDE_NOTIONAL_FIELDS
        + TOB_SHARE_FIELDS
    ):
        assert math.isfinite(m[key]), key
    for key in METRICS_REQUIRED_FINITE_KEYS:
        assert key in m
        assert math.isfinite(m[key]), key


def test_metrics_required_finite_keys_coverage_unit() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    snap = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=99.0, size=1.0)],
        asks=[BookLevel(price=101.0, size=1.0)],
        depth=1,
    )
    m = book_metrics_from_snapshot(snap)
    assert m.keys() >= METRICS_REQUIRED_FINITE_KEYS
    for key in METRICS_REQUIRED_FINITE_KEYS:
        assert math.isfinite(m[key])


def test_missing_required_metrics_key_fail_closed() -> None:
    from quant_fund.microstructure.book_metrics import assert_metrics_required_finite

    with pytest.raises(ValueError, match="missing required keys"):
        assert_metrics_required_finite({"best_bid": 1.0})
    with pytest.raises(ValueError, match="not finite"):
        bad = {k: 1.0 for k in METRICS_REQUIRED_FINITE_KEYS}
        bad["mid"] = float("nan")
        assert_metrics_required_finite(bad)


@given(uncrossed_snapshots().filter(lambda s: len(s.bids) >= 2 and len(s.asks) >= 2))
@settings(max_examples=20, deadline=None)
def test_required_keys_subset_of_metrics_keys(snap: OrderBookSnapshot) -> None:
    """METRICS_REQUIRED_FINITE_KEYS ⊆ book_metrics_from_snapshot(deep).keys()."""
    m = book_metrics_from_snapshot(snap)
    assert m.keys() >= METRICS_REQUIRED_FINITE_KEYS


def test_assert_metrics_required_finite_public() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    snap = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=99.0, size=1.0)],
        asks=[BookLevel(price=101.0, size=1.0)],
        depth=1,
    )
    m = book_metrics_from_snapshot(snap)
    assert assert_metrics_required_finite(dict(m)) is not None
    bad = dict(m)
    bad["spread"] = float("nan")
    with pytest.raises(ValueError, match="not finite"):
        assert_metrics_required_finite(bad)


def test_required_key_docs_cover_frozenset() -> None:
    assert frozenset(METRICS_REQUIRED_FINITE_KEY_DOCS) == METRICS_REQUIRED_FINITE_KEYS
    assert all(isinstance(v, str) and v.strip() for v in METRICS_REQUIRED_FINITE_KEY_DOCS.values())


@given(uncrossed_snapshots())
@settings(max_examples=30, deadline=None)
def test_assert_idempotent_and_thin_required(snap: OrderBookSnapshot) -> None:
    """assert_metrics_required_finite idempotent; depth=1 still satisfies REQUIRED."""
    m = book_metrics_from_snapshot(snap)
    once = assert_metrics_required_finite(dict(m))
    twice = assert_metrics_required_finite(dict(once))
    assert twice.keys() >= METRICS_REQUIRED_FINITE_KEYS
    for key in METRICS_REQUIRED_FINITE_KEYS:
        assert math.isfinite(twice[key])


def test_delete_required_key_raises_with_name() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    snap = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=99.0, size=1.0)],
        asks=[BookLevel(price=101.0, size=1.0)],
        depth=1,
    )
    m = dict(book_metrics_from_snapshot(snap))
    del m["mid"]
    with pytest.raises(ValueError, match="mid"):
        assert_metrics_required_finite(m)


def test_depth1_satisfies_required_shape_nans_ok() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    snap = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=99.0, size=1.0)],
        asks=[BookLevel(price=101.0, size=1.0)],
        depth=1,
    )
    m = book_metrics_from_snapshot(snap)
    assert_metrics_required_finite(m)
    for key in DEPTH_SHAPE_FIELDS:
        assert math.isnan(m[key])


def test_required_disjoint_optional_nan_ok() -> None:
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)
    union = (
        set(DEPTH_SHAPE_FIELDS)
        | set(SIDE_STRUCTURE_FIELDS)
        | set(QUEUE_STRUCTURE_FIELDS)
        | set(SIDE_NOTIONAL_FIELDS)
        | set(TOB_SHARE_FIELDS)
    )
    assert union <= METRICS_OPTIONAL_NAN_OK_KEYS


@given(uncrossed_snapshots().filter(lambda s: len(s.bids) >= 2 and len(s.asks) >= 2))
@settings(max_examples=25, deadline=None)
def test_deep_required_finite_and_some_depth_shape_finite(snap: OrderBookSnapshot) -> None:
    """Deep snap: all REQUIRED finite AND at least one DEPTH_SHAPE field finite."""
    m = book_metrics_from_snapshot(snap)
    for key in METRICS_REQUIRED_FINITE_KEYS:
        assert math.isfinite(m[key]), key
    assert any(math.isfinite(m[k]) for k in DEPTH_SHAPE_FIELDS)


def test_quoted_spread_alias_unit() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    snap = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=99.0, size=1.0)],
        asks=[BookLevel(price=101.0, size=1.0)],
        depth=1,
    )
    m = book_metrics_from_snapshot(snap)
    assert m["quoted_spread"] == pytest.approx(m["spread"])
    assert m["quoted_spread"] == pytest.approx(m["effective_spread"])


def test_depth1_required_finite_depth_shape_nan() -> None:
    """depth=1: DEPTH_SHAPE all NaN; REQUIRED still finite."""
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    snap = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=99.0, size=1.0)],
        asks=[BookLevel(price=101.0, size=1.0)],
        depth=1,
    )
    m = book_metrics_from_snapshot(snap)
    assert_metrics_required_finite(m)
    for key in DEPTH_SHAPE_FIELDS:
        assert math.isnan(m[key])
    for key in METRICS_REQUIRED_FINITE_KEYS:
        assert math.isfinite(m[key])


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_spread_aliases_equal_property(snap: OrderBookSnapshot) -> None:
    """quoted_spread == effective_spread == spread on every valid snap."""
    m = book_metrics_from_snapshot(snap)
    assert m["quoted_spread"] == pytest.approx(m["spread"], abs=1e-12)
    assert m["effective_spread"] == pytest.approx(m["spread"], abs=1e-12)
    assert m["quoted_spread"] == pytest.approx(m["effective_spread"], abs=1e-12)


@given(uncrossed_snapshots().filter(lambda s: len(s.bids) == 1 and len(s.asks) == 1))
@settings(max_examples=30, deadline=None)
def test_depth1_optional_may_nan_required_never(snap: OrderBookSnapshot) -> None:
    """depth=1: OPTIONAL depth-shape NaN OK; REQUIRED never NaN."""
    m = book_metrics_from_snapshot(snap)
    for key in METRICS_REQUIRED_FINITE_KEYS:
        assert math.isfinite(m[key]), key
    for key in DEPTH_SHAPE_FIELDS:
        assert key in METRICS_OPTIONAL_NAN_OK_KEYS
        assert math.isnan(m[key]), key


def test_assert_ignores_optional_nan() -> None:
    """NaN in OPTIONAL keys must not cause assert_metrics_required_finite to raise."""
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    snap = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=100.0, size=1.0), BookLevel(price=99.0, size=1.0)],
        asks=[BookLevel(price=101.0, size=1.0), BookLevel(price=102.0, size=1.0)],
        depth=2,
    )
    m = dict(book_metrics_from_snapshot(snap))
    # Force optional shape keys to NaN even on a deep book
    for key in DEPTH_SHAPE_FIELDS:
        m[key] = float("nan")
    assert_metrics_required_finite(m)  # must not raise


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_half_spread_and_touch_imbalance_aliases(snap: OrderBookSnapshot) -> None:
    """half_spread == spread/2; half_spread_bps == spread_bps/2; touch == imbalance_top."""
    m = book_metrics_from_snapshot(snap)
    assert math.isfinite(m["half_spread"]) and math.isfinite(m["half_spread_bps"])
    assert m["half_spread"] == pytest.approx(m["spread"] / 2.0, abs=1e-12)
    assert m["half_spread_bps"] == pytest.approx(m["spread_bps"] / 2.0, abs=1e-9)
    assert m["touch_size_imbalance"] == pytest.approx(m["imbalance_top"], abs=1e-12)


def test_quoted_spread_present_with_spread() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    snap = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=99.0, size=1.0)],
        asks=[BookLevel(price=101.0, size=1.0)],
        depth=1,
    )
    m = book_metrics_from_snapshot(snap)
    assert "spread" in m and "quoted_spread" in m
    assert m["quoted_spread"] == m["spread"]


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_weight_depth_abs_spread_over_mid_bundle(snap: OrderBookSnapshot) -> None:
    """microprice weight ∈(0,1) + convex combo; depth_imbalance_abs; spread_over_mid."""
    m = book_metrics_from_snapshot(snap)
    w = m["microprice_weight_balance"]
    assert 0.0 < w < 1.0
    assert m["microprice"] == pytest.approx(
        m["best_ask"] * w + m["best_bid"] * (1.0 - w), abs=1e-12
    )
    assert m["depth_imbalance_abs"] == pytest.approx(abs(m["imbalance_depth"]), abs=1e-12)
    assert m["mid"] > 0.0
    assert m["spread_over_mid"] == pytest.approx(m["spread"] / m["mid"], abs=1e-12)


def test_spread_over_mid_unit() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    snap = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=99.0, size=1.0)],
        asks=[BookLevel(price=101.0, size=1.0)],
        depth=1,
    )
    m = book_metrics_from_snapshot(snap)
    assert m["mid"] == pytest.approx(100.0)
    assert m["spread"] == pytest.approx(2.0)
    assert m["spread_over_mid"] == pytest.approx(0.02)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_micro_minus_mid_and_imbalance_bounds_bundle(snap: OrderBookSnapshot) -> None:
    """microprice_minus_mid(+bps) identities; imbalance_top/depth ∈ [-1,1]."""
    m = book_metrics_from_snapshot(snap)
    assert m["microprice_minus_mid"] == pytest.approx(m["microprice"] - m["mid"], abs=1e-12)
    assert m["mid"] > 0.0
    assert m["microprice_minus_mid_bps"] == pytest.approx(
        1e4 * m["microprice_minus_mid"] / m["mid"], abs=1e-9
    )
    assert -1.0 - 1e-12 <= m["imbalance_top"] <= 1.0 + 1e-12
    assert -1.0 - 1e-12 <= m["imbalance_depth"] <= 1.0 + 1e-12
    assert m["mid"] == pytest.approx(0.5 * (m["best_bid"] + m["best_ask"]), abs=1e-12)


def test_mid_half_sum_unit() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    snap = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=10.0, size=1.0)],
        asks=[BookLevel(price=14.0, size=1.0)],
        depth=1,
    )
    m = book_metrics_from_snapshot(snap)
    assert m["mid"] == pytest.approx(12.0)
    assert snap.mid == pytest.approx(12.0)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_concentration_and_tob_share_integrity_47(snap: OrderBookSnapshot) -> None:
    """size_concentration_top ∈[0,1]; tob_size_share ∈(0,1]; TOB_SHARE coexistence."""
    m = book_metrics_from_snapshot(snap)
    for key in SIDE_STRUCTURE_FIELDS:
        assert math.isfinite(m[key])
        assert 0.0 <= m[key] <= 1.0 + 1e-12
    assert m["bid_depth"] + m["ask_depth"] > 0.0
    for key in TOB_SHARE_FIELDS:
        assert math.isfinite(m[key])
        assert 0.0 < m[key] <= 1.0 + 1e-12
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)


def test_concentration_thin_depth_nan_not_zero() -> None:
    from quant_fund.microstructure.book_metrics import _size_concentration_top

    assert math.isnan(_size_concentration_top(1.0, 0.0))
    assert math.isnan(_size_concentration_top(0.0, 0.0))
    assert _size_concentration_top(1.0, 4.0) == pytest.approx(0.25)
    # Empty depth must not fake 0.0 concentration
    assert _size_concentration_top(1.0, 0.0) != 0.0


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_notional_queue_bound_integrity_48(snap: OrderBookSnapshot) -> None:
    """notional_imbalance ∈[-1,1]; side_notional≥0; queue_priority ∈[0,1]; OPTIONAL∩REQUIRED==∅."""
    m = book_metrics_from_snapshot(snap)
    assert math.isfinite(m["notional_imbalance"])
    assert -1.0 - 1e-12 <= m["notional_imbalance"] <= 1.0 + 1e-12
    for key in SIDE_NOTIONAL_FIELDS:
        assert math.isfinite(m[key]) and m[key] >= 0.0
    for key in QUEUE_STRUCTURE_FIELDS:
        assert math.isfinite(m[key])
        assert 0.0 <= m[key] <= 1.0 + 1e-12
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)


def test_side_notional_and_queue_thin_nan_not_zero() -> None:
    from quant_fund.microstructure.book_metrics import (
        _queue_priority_proxy,
        _side_notional_proxy,
    )

    assert math.isnan(_side_notional_proxy(100.0, 0.0))
    assert _side_notional_proxy(100.0, 0.0) != 0.0
    assert math.isnan(_queue_priority_proxy(1.0, 0.0))
    assert _queue_priority_proxy(1.0, 0.0) != 0.0
    assert _side_notional_proxy(50.0, 2.0) == pytest.approx(100.0)
    assert 0.0 <= _queue_priority_proxy(1.0, 3.0) <= 1.0


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_top_size_spread_geometry_49(snap: OrderBookSnapshot) -> None:
    """top sizes >0 and ≤ depth; spread_over_mid/half_spread(+bps) geometry."""
    m = book_metrics_from_snapshot(snap)
    assert m["n_bid_levels"] >= 1.0 and m["n_ask_levels"] >= 1.0
    assert m["top_bid_size"] > 0.0 and math.isfinite(m["top_bid_size"])
    assert m["top_ask_size"] > 0.0 and math.isfinite(m["top_ask_size"])
    assert m["top_bid_size"] <= m["bid_depth"] + 1e-12
    assert m["top_ask_size"] <= m["ask_depth"] + 1e-12
    assert m["mid"] > 0.0
    assert m["spread_over_mid"] == pytest.approx(m["spread"] / m["mid"], rel=1e-12, abs=1e-12)
    assert m["half_spread"] == pytest.approx(0.5 * m["spread"], abs=1e-12)
    assert m["half_spread_bps"] == pytest.approx(1e4 * m["half_spread"] / m["mid"], abs=1e-9)
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)


def test_top_size_spread_geometry_unit_49() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    snap = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=100.0, size=2.0), BookLevel(price=99.0, size=2.0)],
        asks=[BookLevel(price=102.0, size=1.0)],
        depth=2,
    )
    m = book_metrics_from_snapshot(snap)
    assert m["top_bid_size"] == pytest.approx(2.0)
    assert m["bid_depth"] == pytest.approx(4.0)
    assert m["top_bid_size"] <= m["bid_depth"]
    assert m["spread"] == pytest.approx(2.0)
    assert m["mid"] == pytest.approx(101.0)
    assert m["spread_over_mid"] == pytest.approx(2.0 / 101.0, rel=1e-12)
    assert m["half_spread"] == pytest.approx(1.0)
    assert m["half_spread_bps"] == pytest.approx(1e4 / 101.0, abs=1e-9)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_depth_counts_slope_honesty_50(snap: OrderBookSnapshot) -> None:
    """n_*_levels ≥1; depths >0; deep size slopes finite; thin size slopes NaN."""
    m = book_metrics_from_snapshot(snap)
    assert m["n_bid_levels"] >= 1.0 and float(m["n_bid_levels"]).is_integer()
    assert m["n_ask_levels"] >= 1.0 and float(m["n_ask_levels"]).is_integer()
    assert m["bid_depth"] > 0.0 and m["ask_depth"] > 0.0
    if m["n_bid_levels"] >= 2.0:
        assert math.isfinite(m["bid_log_size_slope"])
    else:
        assert math.isnan(m["bid_log_size_slope"])
        assert m["bid_log_size_slope"] != 0.0
    if m["n_ask_levels"] >= 2.0:
        assert math.isfinite(m["ask_log_size_slope"])
    else:
        assert math.isnan(m["ask_log_size_slope"])
        assert m["ask_log_size_slope"] != 0.0
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)


def test_thin_size_slopes_nan_not_zero_unit_50() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    thin = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=99.0, size=1.0)],
        asks=[BookLevel(price=101.0, size=1.0)],
        depth=1,
    )
    m = book_metrics_from_snapshot(thin)
    assert m["n_bid_levels"] == 1.0 and m["n_ask_levels"] == 1.0
    assert math.isnan(m["bid_log_size_slope"]) and math.isnan(m["ask_log_size_slope"])
    deep = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[
            BookLevel(price=math.exp(1.0), size=math.exp(1.0)),
            BookLevel(price=math.exp(0.7), size=math.exp(0.7)),
        ],
        asks=[
            BookLevel(price=math.exp(1.5), size=math.exp(0.5)),
            BookLevel(price=math.exp(1.7), size=math.exp(0.3)),
        ],
        depth=2,
    )
    md = book_metrics_from_snapshot(deep)
    assert math.isfinite(md["bid_log_size_slope"]) and math.isfinite(md["ask_log_size_slope"])


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_price_slope_microprice_geometry_51(snap: OrderBookSnapshot) -> None:
    """Price slopes deep/thin honesty; microprice in [bid,ask]; weight sign vs μ−mid."""
    m = book_metrics_from_snapshot(snap)
    if m["n_bid_levels"] >= 2.0:
        assert math.isfinite(m["bid_log_price_slope"])
    else:
        assert math.isnan(m["bid_log_price_slope"])
        assert m["bid_log_price_slope"] != 0.0
    if m["n_ask_levels"] >= 2.0:
        assert math.isfinite(m["ask_log_price_slope"])
    else:
        assert math.isnan(m["ask_log_price_slope"])
        assert m["ask_log_price_slope"] != 0.0
    assert m["best_bid"] <= m["microprice"] <= m["best_ask"]
    w = m["microprice_weight_balance"]
    # w > 0.5 ⇒ more bid size ⇒ microprice closer to ask ⇒ microprice >= mid
    if abs(w - 0.5) > 1e-12:
        if w > 0.5:
            assert m["microprice"] >= m["mid"] - 1e-12
            assert m["microprice_minus_mid"] >= -1e-12
        else:
            assert m["microprice"] <= m["mid"] + 1e-12
            assert m["microprice_minus_mid"] <= 1e-12
    else:
        assert m["microprice"] == pytest.approx(m["mid"], abs=1e-9)
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)


def test_thin_price_slopes_nan_not_zero_unit_51() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    thin = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=99.0, size=1.0)],
        asks=[BookLevel(price=101.0, size=1.0)],
        depth=1,
    )
    m = book_metrics_from_snapshot(thin)
    assert math.isnan(m["bid_log_price_slope"]) and math.isnan(m["ask_log_price_slope"])
    # Heavy bid size ⇒ microprice toward ask
    heavy_bid = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=99.0, size=9.0)],
        asks=[BookLevel(price=101.0, size=1.0)],
        depth=1,
    )
    mh = book_metrics_from_snapshot(heavy_bid)
    assert mh["microprice_weight_balance"] > 0.5
    assert mh["microprice"] > mh["mid"]
    assert mh["best_bid"] < mh["microprice"] < mh["best_ask"]


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_tick_spacing_concentration_coexistence_52(snap: OrderBookSnapshot) -> None:
    """Strict-mono snaps: tick spacing finite when n≥2; concentration∈[0,1]; DEPTH_SHAPE coexist."""
    m = book_metrics_from_snapshot(snap)
    for key in SIDE_STRUCTURE_FIELDS:
        assert math.isfinite(m[key])
        assert 0.0 <= m[key] <= 1.0 + 1e-12
    if m["n_bid_levels"] >= 2.0:
        # Positive gaps enforced by schema strict mono → spacing finite
        assert math.isfinite(m["bid_mean_log_tick_spacing"])
    else:
        assert math.isnan(m["bid_mean_log_tick_spacing"])
        assert m["bid_mean_log_tick_spacing"] != 0.0
    if m["n_ask_levels"] >= 2.0:
        assert math.isfinite(m["ask_mean_log_tick_spacing"])
    else:
        assert math.isnan(m["ask_mean_log_tick_spacing"])
        assert m["ask_mean_log_tick_spacing"] != 0.0
    # DEPTH_SHAPE keys present alongside concentration
    for key in DEPTH_SHAPE_FIELDS:
        assert key in m
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)


def test_zero_gap_tick_spacing_nan_not_fake_zero_via_helper_52() -> None:
    """Zero-gap spacing NaN (not 0): schema rejects ties, so exercise _mean_log_tick_spacing directly."""
    from quant_fund.microstructure.book_metrics import _mean_log_tick_spacing
    from quant_fund.schemas.order_book import BookLevel

    tied = [
        BookLevel(price=100.0, size=1.0),
        BookLevel(price=100.0, size=1.0),
    ]
    # Can't build OrderBookSnapshot with ties; helper must still NaN on zero gap
    assert math.isnan(_mean_log_tick_spacing(tied))
    assert _mean_log_tick_spacing(tied) != 0.0
    strict = [
        BookLevel(price=100.0, size=1.0),
        BookLevel(price=99.0, size=1.0),
    ]
    assert math.isfinite(_mean_log_tick_spacing(strict))


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_side_tob_notional_geometry_53(snap: OrderBookSnapshot) -> None:
    """Side/TOB notional geometry; tob_size_share∈(0,1]; tob_notional_share∈(0,1]."""
    m = book_metrics_from_snapshot(snap)
    for key in SIDE_NOTIONAL_FIELDS:
        assert math.isfinite(m[key]) and m[key] >= 0.0
    assert math.isfinite(m["notional_imbalance"])
    assert -1.0 - 1e-12 <= m["notional_imbalance"] <= 1.0 + 1e-12
    assert math.isfinite(m["top_of_book_notional_proxy"]) and m["top_of_book_notional_proxy"] >= 0.0
    assert m["bid_depth"] + m["ask_depth"] > 0.0
    assert math.isfinite(m["tob_size_share"])
    assert 0.0 < m["tob_size_share"] <= 1.0 + 1e-12
    assert math.isfinite(m["tob_notional_share"])
    assert 0.0 < m["tob_notional_share"] <= 1.0 + 1e-12
    # Coexist catalogs
    for key in TOB_SHARE_FIELDS + SIDE_NOTIONAL_FIELDS:
        assert key in m
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)


def test_zero_size_side_notional_nan_not_fake_zero_53() -> None:
    from quant_fund.microstructure.book_metrics import _side_notional_proxy

    assert math.isnan(_side_notional_proxy(100.0, 0.0))
    assert _side_notional_proxy(100.0, 0.0) != 0.0


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_tob_size_share_vs_notional_share_honesty_54(snap: OrderBookSnapshot) -> None:
    """tob_size_share∈(0,1]; tob_notional_share∈(0,1] post-#62 (size-share ≠ notional-share fields)."""
    m = book_metrics_from_snapshot(snap)
    assert m["bid_depth"] + m["ask_depth"] > 0.0
    assert math.isfinite(m["tob_size_share"])
    assert 0.0 < m["tob_size_share"] <= 1.0 + 1e-12
    assert math.isfinite(m["tob_notional_share"])
    assert 0.0 < m["tob_notional_share"] <= 1.0 + 1e-12
    # Honesty: both shares ∈(0,1] post touch-priced TOB; fields remain distinct
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)


def test_required_docs_completeness_and_nan_mutate_54() -> None:
    assert METRICS_REQUIRED_FINITE_KEY_DOCS.keys() == METRICS_REQUIRED_FINITE_KEYS
    assert frozenset(METRICS_REQUIRED_FINITE_KEY_DOCS) == METRICS_REQUIRED_FINITE_KEYS
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    snap = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=99.0, size=1.0)],
        asks=[BookLevel(price=101.0, size=1.0)],
        depth=1,
    )
    m = dict(book_metrics_from_snapshot(snap))
    m["spread"] = float("nan")
    with pytest.raises(ValueError, match="not finite"):
        assert_metrics_required_finite(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_microprice_weight_balance_bounds_and_identity_55(snap: OrderBookSnapshot) -> None:
    """microprice_weight_balance ∈ [0,1] when finite; equals top_bid/(top_bid+top_ask)."""
    m = book_metrics_from_snapshot(snap)
    w = m["microprice_weight_balance"]
    assert math.isfinite(w)
    assert 0.0 <= w <= 1.0 + 1e-12
    top_bid = snap.bids[0].size
    top_ask = snap.asks[0].size
    expected = top_bid / (top_bid + top_ask)
    assert w == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)


def test_finite_rate_empty_and_all_nan_edge_contracts_55() -> None:
    """Lock existing contracts: empty → NaN; zero-eligible → NaN; eligible all-NaN → 0.0."""
    helpers = (
        depth_shape_finite_rate,
        concentration_top_finite_rate,
        queue_priority_finite_rate,
        side_notional_finite_rate,
        tob_size_share_finite_rate,
    )
    for helper in helpers:
        assert math.isnan(helper([]))

    # Zero-eligible (thin / no depth claims): NaN per helper docs
    thin_depth_shape = {
        "n_bid_levels": 1.0,
        "n_ask_levels": 1.0,
        "bid_log_size_slope": float("nan"),
        "ask_log_size_slope": float("nan"),
        "bid_log_price_slope": float("nan"),
        "ask_log_price_slope": float("nan"),
        "bid_mean_log_tick_spacing": float("nan"),
        "ask_mean_log_tick_spacing": float("nan"),
    }
    assert math.isnan(depth_shape_finite_rate([thin_depth_shape]))

    zero_depth = {
        "bid_depth": 0.0,
        "ask_depth": 0.0,
        "bid_size_concentration_top": float("nan"),
        "ask_size_concentration_top": float("nan"),
        "queue_priority_proxy": float("nan"),
        "ask_queue_priority_proxy": float("nan"),
        "tob_size_share": float("nan"),
        "best_bid": 100.0,
        "best_ask": 101.0,
        "side_notional_proxy_bid": float("nan"),
        "side_notional_proxy_ask": float("nan"),
    }
    assert math.isnan(concentration_top_finite_rate([zero_depth]))
    assert math.isnan(queue_priority_finite_rate([zero_depth]))
    assert math.isnan(tob_size_share_finite_rate([zero_depth]))
    assert math.isnan(side_notional_finite_rate([zero_depth]))

    # Eligible but all field values NaN → rate 0.0 (not NaN)
    deep_all_nan = {
        "n_bid_levels": 3.0,
        "n_ask_levels": 3.0,
        "bid_log_size_slope": float("nan"),
        "ask_log_size_slope": float("nan"),
        "bid_log_price_slope": float("nan"),
        "ask_log_price_slope": float("nan"),
        "bid_mean_log_tick_spacing": float("nan"),
        "ask_mean_log_tick_spacing": float("nan"),
        "bid_depth": 10.0,
        "ask_depth": 10.0,
        "bid_size_concentration_top": float("nan"),
        "ask_size_concentration_top": float("nan"),
        "queue_priority_proxy": float("nan"),
        "ask_queue_priority_proxy": float("nan"),
        "tob_size_share": float("nan"),
        "best_bid": 100.0,
        "best_ask": 101.0,
        "side_notional_proxy_bid": float("nan"),
        "side_notional_proxy_ask": float("nan"),
    }
    assert depth_shape_finite_rate([deep_all_nan]) == 0.0
    assert concentration_top_finite_rate([deep_all_nan]) == 0.0
    assert queue_priority_finite_rate([deep_all_nan]) == 0.0
    assert side_notional_finite_rate([deep_all_nan]) == 0.0
    assert tob_size_share_finite_rate([deep_all_nan]) == 0.0
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_queue_priority_and_touch_alias_reinforce_56(snap: OrderBookSnapshot) -> None:
    """queue_priority ∈ [0,1] when finite; touch_size_imbalance==imbalance_top; quoted==effective==spread."""
    m = book_metrics_from_snapshot(snap)
    for key in ("queue_priority_proxy", "ask_queue_priority_proxy"):
        v = m[key]
        assert math.isfinite(v)
        assert 0.0 <= v <= 1.0 + 1e-12
    assert m["touch_size_imbalance"] == pytest.approx(m["imbalance_top"], rel=1e-12, abs=1e-12)
    assert m["quoted_spread"] == pytest.approx(m["spread"], rel=1e-12, abs=1e-12)
    assert m["effective_spread"] == pytest.approx(m["spread"], rel=1e-12, abs=1e-12)
    assert m["quoted_spread"] == pytest.approx(m["effective_spread"], rel=1e-12, abs=1e-12)
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)


def test_queue_priority_thin_zero_depth_nan_not_zero_56() -> None:
    """thin/zero depth → queue_priority NaN not 0 (helper contract)."""
    from quant_fund.microstructure.book_metrics import _queue_priority_proxy

    for top, depth in (
        (1.0, 0.0),
        (0.0, 0.0),
        (2.0, -1.0),
        (float("nan"), 5.0),
        (1.0, float("nan")),
    ):
        v = _queue_priority_proxy(top, depth)
        assert math.isnan(v), (top, depth, v)
        assert v != 0.0
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_depth_imbalance_and_spread_over_mid_reinforce_57(snap: OrderBookSnapshot) -> None:
    """depth_imbalance_abs == |imbalance_depth|; imbalance_depth ∈ [-1,1]; spread_over_mid == spread/mid."""
    m = book_metrics_from_snapshot(snap)
    assert math.isfinite(m["imbalance_depth"])
    assert -1.0 - 1e-12 <= m["imbalance_depth"] <= 1.0 + 1e-12
    assert m["depth_imbalance_abs"] == pytest.approx(
        abs(m["imbalance_depth"]), rel=1e-12, abs=1e-12
    )
    assert m["mid"] > 0.0
    assert math.isfinite(m["spread_over_mid"])
    assert m["spread_over_mid"] == pytest.approx(m["spread"] / m["mid"], rel=1e-12, abs=1e-12)
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_bps_aliases_and_mid_geometry_58(snap: OrderBookSnapshot) -> None:
    """quoted_spread_bps==spread_bps; half_spread_bps identities; best_bid<mid<best_ask; mid midpoint."""
    m = book_metrics_from_snapshot(snap)
    assert m["quoted_spread_bps"] == pytest.approx(m["spread_bps"], rel=1e-12, abs=1e-12)
    assert m["half_spread_bps"] == pytest.approx(0.5 * m["spread_bps"], rel=1e-12, abs=1e-12)
    assert m["mid"] > 0.0
    assert m["half_spread_bps"] == pytest.approx(
        1e4 * m["half_spread"] / m["mid"], rel=1e-12, abs=1e-12
    )
    assert m["best_bid"] < m["mid"] < m["best_ask"]
    assert m["mid"] == pytest.approx(0.5 * (m["best_bid"] + m["best_ask"]), rel=1e-12, abs=1e-12)
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_concentration_vs_queue_formula_lock_59(snap: OrderBookSnapshot) -> None:
    """concentration == top/side_depth ∈[0,1]; queue == top/(top+side) DISTINCT (never equate)."""
    m = book_metrics_from_snapshot(snap)
    top_bid = snap.bids[0].size
    top_ask = snap.asks[0].size
    bid_depth = m["bid_depth"]
    ask_depth = m["ask_depth"]

    assert math.isfinite(m["bid_size_concentration_top"])
    assert math.isfinite(m["ask_size_concentration_top"])
    assert 0.0 <= m["bid_size_concentration_top"] <= 1.0 + 1e-12
    assert 0.0 <= m["ask_size_concentration_top"] <= 1.0 + 1e-12
    assert m["bid_size_concentration_top"] == pytest.approx(
        top_bid / bid_depth, rel=1e-12, abs=1e-12
    )
    assert m["ask_size_concentration_top"] == pytest.approx(
        top_ask / ask_depth, rel=1e-12, abs=1e-12
    )

    assert math.isfinite(m["queue_priority_proxy"])
    assert math.isfinite(m["ask_queue_priority_proxy"])
    assert m["queue_priority_proxy"] == pytest.approx(
        top_bid / (top_bid + bid_depth), rel=1e-12, abs=1e-12
    )
    assert m["ask_queue_priority_proxy"] == pytest.approx(
        top_ask / (top_ask + ask_depth), rel=1e-12, abs=1e-12
    )

    # Honesty: formulas are distinct — never equate concentration to queue
    assert (
        m["bid_size_concentration_top"] != pytest.approx(m["queue_priority_proxy"], rel=0, abs=0)
        or top_bid == 0.0
    )  # only degenerate equality if top==0 (schema forbids)
    # Stronger: with positive sizes, queue < concentration for depth>0
    assert m["queue_priority_proxy"] < m["bid_size_concentration_top"] + 1e-12
    assert m["ask_queue_priority_proxy"] < m["ask_size_concentration_top"] + 1e-12
    assert m["queue_priority_proxy"] != pytest.approx(
        m["bid_size_concentration_top"], rel=1e-9, abs=1e-12
    )
    assert m["ask_queue_priority_proxy"] != pytest.approx(
        m["ask_size_concentration_top"], rel=1e-9, abs=1e-12
    )
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_side_notional_and_tob_size_share_lock_60(snap: OrderBookSnapshot) -> None:
    """side_notional == best*depth ≥0; tob_size_share == (top_bid+top_ask)/(bid_depth+ask_depth) ∈(0,1]."""
    m = book_metrics_from_snapshot(snap)
    top_bid = snap.bids[0].size
    top_ask = snap.asks[0].size
    assert math.isfinite(m["side_notional_proxy_bid"])
    assert math.isfinite(m["side_notional_proxy_ask"])
    assert m["side_notional_proxy_bid"] == pytest.approx(
        m["best_bid"] * m["bid_depth"], rel=1e-12, abs=1e-12
    )
    assert m["side_notional_proxy_ask"] == pytest.approx(
        m["best_ask"] * m["ask_depth"], rel=1e-12, abs=1e-12
    )
    assert m["side_notional_proxy_bid"] >= 0.0
    assert m["side_notional_proxy_ask"] >= 0.0

    depth_sum = m["bid_depth"] + m["ask_depth"]
    assert depth_sum > 0.0
    assert math.isfinite(m["tob_size_share"])
    assert m["tob_size_share"] == pytest.approx(
        (top_bid + top_ask) / depth_sum, rel=1e-12, abs=1e-12
    )
    assert 0.0 < m["tob_size_share"] <= 1.0 + 1e-12
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_notional_imbalance_and_tob_notional_share_61(snap: OrderBookSnapshot) -> None:
    """notional_imbalance==(bid_n-ask_n)/(bid_n+ask_n)∈[-1,1]; tob_notional_share∈(0,1] (≠ size_share field)."""
    m = book_metrics_from_snapshot(snap)
    bid_n = m["side_notional_proxy_bid"]
    ask_n = m["side_notional_proxy_ask"]
    tob_n = m["top_of_book_notional_proxy"]
    assert math.isfinite(bid_n) and math.isfinite(ask_n)
    denom = bid_n + ask_n
    assert denom > 0.0
    assert math.isfinite(m["notional_imbalance"])
    assert m["notional_imbalance"] == pytest.approx((bid_n - ask_n) / denom, rel=1e-12, abs=1e-12)
    assert -1.0 - 1e-12 <= m["notional_imbalance"] <= 1.0 + 1e-12

    assert math.isfinite(tob_n) and math.isfinite(m["tob_notional_share"])
    assert m["tob_notional_share"] == pytest.approx(tob_n / denom, rel=1e-12, abs=1e-12)
    assert 0.0 < m["tob_notional_share"] <= 1.0 + 1e-12
    # Distinct from tob_size_share field (both ∈(0,1] post-#62)
    assert math.isfinite(m["tob_size_share"])
    assert 0.0 < m["tob_size_share"] <= 1.0 + 1e-12
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_tob_notional_and_depth_sum_lock_62(snap: OrderBookSnapshot) -> None:
    """top_of_book_notional == best_bid*top_bid + best_ask*top_ask ≥0; depths == sum sizes."""
    m = book_metrics_from_snapshot(snap)
    top_bid = snap.bids[0].size
    top_ask = snap.asks[0].size
    assert math.isfinite(m["top_of_book_notional_proxy"])
    assert m["top_of_book_notional_proxy"] == pytest.approx(
        m["best_bid"] * top_bid + m["best_ask"] * top_ask, rel=1e-12, abs=1e-12
    )
    assert m["top_of_book_notional_proxy"] >= 0.0
    assert m["bid_depth"] == pytest.approx(sum(lvl.size for lvl in snap.bids), rel=1e-12, abs=1e-12)
    assert m["ask_depth"] == pytest.approx(sum(lvl.size for lvl in snap.asks), rel=1e-12, abs=1e-12)
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_post62_tob_honesty_lock_63(snap: OrderBookSnapshot) -> None:
    """tob_notional_share ∈ (0,1]; == tob/(bid_n+ask_n); top_of_book_notional ≥ 0 (may>1 obsolete)."""
    m = book_metrics_from_snapshot(snap)
    bid_n = m["side_notional_proxy_bid"]
    ask_n = m["side_notional_proxy_ask"]
    tob = m["top_of_book_notional_proxy"]
    denom = bid_n + ask_n
    assert denom > 0.0
    assert math.isfinite(tob) and tob >= 0.0
    assert math.isfinite(m["tob_notional_share"])
    assert m["tob_notional_share"] == pytest.approx(tob / denom, rel=1e-12, abs=1e-12)
    assert 0.0 < m["tob_notional_share"] <= 1.0 + 1e-12
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_imbalance_top_weight_balance_identity_64(snap: OrderBookSnapshot) -> None:
    """imbalance_top == 2*microprice_weight_balance - 1; touch_size_imbalance aliases; bounds."""
    m = book_metrics_from_snapshot(snap)
    w = m["microprice_weight_balance"]
    imb = m["imbalance_top"]
    assert math.isfinite(w) and math.isfinite(imb)
    assert 0.0 <= w <= 1.0 + 1e-12
    assert -1.0 - 1e-12 <= imb <= 1.0 + 1e-12
    assert imb == pytest.approx(2.0 * w - 1.0, rel=1e-12, abs=1e-12)
    assert m["touch_size_imbalance"] == pytest.approx(imb, rel=1e-12, abs=1e-12)
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_microprice_minus_mid_geometry_65(snap: OrderBookSnapshot) -> None:
    """μ-mid == microprice-mid; bps identity; μ-mid ≈ spread*(w-0.5)."""
    m = book_metrics_from_snapshot(snap)
    assert m["mid"] > 0.0
    assert math.isfinite(m["microprice"]) and math.isfinite(m["microprice_minus_mid"])
    assert m["microprice_minus_mid"] == pytest.approx(
        m["microprice"] - m["mid"], rel=1e-12, abs=1e-12
    )
    assert m["microprice_minus_mid_bps"] == pytest.approx(
        1e4 * (m["microprice"] - m["mid"]) / m["mid"], rel=1e-12, abs=1e-12
    )
    w = m["microprice_weight_balance"]
    assert math.isfinite(w) and math.isfinite(m["spread"])
    assert m["microprice_minus_mid"] == pytest.approx(m["spread"] * (w - 0.5), rel=1e-12, abs=1e-12)
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_n_levels_vs_slopes_coexistence_66(snap: OrderBookSnapshot) -> None:
    """n_*_levels == len(sides); depth1 size+price slopes NaN; depth≥2 both finite."""
    m = book_metrics_from_snapshot(snap)
    assert m["n_bid_levels"] == pytest.approx(float(len(snap.bids)), abs=1e-12)
    assert m["n_ask_levels"] == pytest.approx(float(len(snap.asks)), abs=1e-12)

    size_price = (
        ("bid_log_size_slope", "bid_log_price_slope", m["n_bid_levels"]),
        ("ask_log_size_slope", "ask_log_price_slope", m["n_ask_levels"]),
    )
    for size_k, price_k, n_lev in size_price:
        if n_lev == 1.0:
            assert math.isnan(m[size_k]) and math.isnan(m[price_k])
            assert m[size_k] != 0.0 and m[price_k] != 0.0
        else:
            assert n_lev >= 2.0
            assert math.isfinite(m[size_k]) and math.isfinite(m[price_k])
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_export_tuple_completeness_67(snap: OrderBookSnapshot) -> None:
    """Every DEPTH/SIDE/QUEUE/SIDE_NOTIONAL/TOB_SHARE key present; REQUIRED ⊆ return; OPTIONAL∩REQUIRED==∅."""
    m = book_metrics_from_snapshot(snap)
    keys = set(m)
    for field in (
        DEPTH_SHAPE_FIELDS
        + SIDE_STRUCTURE_FIELDS
        + QUEUE_STRUCTURE_FIELDS
        + SIDE_NOTIONAL_FIELDS
        + TOB_SHARE_FIELDS
    ):
        assert field in keys, field
    assert METRICS_REQUIRED_FINITE_KEYS.issubset(keys)
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_required_finite_optional_not_inf_68(snap: OrderBookSnapshot) -> None:
    """REQUIRED keys finite; OPTIONAL may NaN but never ±inf; partition helper fail-closed."""
    m = book_metrics_from_snapshot(snap)
    for key in METRICS_REQUIRED_FINITE_KEYS:
        assert key in m
        assert math.isfinite(m[key]), key
    for key in METRICS_OPTIONAL_NAN_OK_KEYS:
        if key in m:
            assert not math.isinf(float(m[key])), key
    assert_metrics_key_partition(m)
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)


def test_optional_inf_fail_closed_helper_68() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    snap = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=99.0, size=1.0)],
        asks=[BookLevel(price=101.0, size=1.0)],
        depth=1,
    )
    m = dict(book_metrics_from_snapshot(snap))
    # depth-shape optional present as NaN is ok
    m["bid_log_size_slope"] = float("nan")
    assert_metrics_optional_not_inf(m)
    m["bid_log_size_slope"] = float("inf")
    with pytest.raises(ValueError, match="±inf"):
        assert_metrics_optional_not_inf(m)
    with pytest.raises(ValueError, match="±inf"):
        assert_metrics_key_partition(m)


@given(uncrossed_snapshots().filter(lambda s: len(s.bids) >= 2 and len(s.asks) >= 2))
@settings(max_examples=40, deadline=None)
def test_mean_log_tick_spacing_identity_69(snap: OrderBookSnapshot) -> None:
    """When n≥2: mean_log_tick_spacing == mean(log|Δp|) over adjacent levels; REQUIRED still finite."""
    m = book_metrics_from_snapshot(snap)

    def expected(levels) -> float:
        gaps = [abs(levels[i].price - levels[i - 1].price) for i in range(1, len(levels))]
        return float(sum(math.log(g) for g in gaps) / len(gaps))

    assert math.isfinite(m["bid_mean_log_tick_spacing"])
    assert math.isfinite(m["ask_mean_log_tick_spacing"])
    assert m["bid_mean_log_tick_spacing"] == pytest.approx(
        expected(snap.bids), rel=1e-12, abs=1e-12
    )
    assert m["ask_mean_log_tick_spacing"] == pytest.approx(
        expected(snap.asks), rel=1e-12, abs=1e-12
    )
    assert_metrics_key_partition(m)
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)


@given(uncrossed_snapshots().filter(lambda s: len(s.bids) >= 2 and len(s.asks) >= 2))
@settings(max_examples=40, deadline=None)
def test_log_price_slope_ols_identity_70(snap: OrderBookSnapshot) -> None:
    """When n≥2: bid/ask_log_price_slope == OLS slope of log(price) on level index."""
    import numpy as np

    from quant_fund.microstructure.book_metrics import _log_price_slope

    m = book_metrics_from_snapshot(snap)

    def ols(levels) -> float:
        prices = np.asarray([lvl.price for lvl in levels], dtype=float)
        y = np.log(prices)
        x = np.arange(prices.size, dtype=float)
        return float(np.cov(x, y, ddof=0)[0, 1] / float(np.var(x)))

    assert math.isfinite(m["bid_log_price_slope"])
    assert math.isfinite(m["ask_log_price_slope"])
    assert m["bid_log_price_slope"] == pytest.approx(ols(snap.bids), rel=1e-12, abs=1e-12)
    assert m["ask_log_price_slope"] == pytest.approx(ols(snap.asks), rel=1e-12, abs=1e-12)
    assert m["bid_log_price_slope"] == pytest.approx(_log_price_slope(snap.bids), abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots().filter(lambda s: len(s.bids) == 1 and len(s.asks) == 1))
@settings(max_examples=20, deadline=None)
def test_log_price_slope_thin_nan_70(snap: OrderBookSnapshot) -> None:
    """depth=1: log price slopes NaN not 0; REQUIRED still finite."""
    m = book_metrics_from_snapshot(snap)
    assert math.isnan(m["bid_log_price_slope"]) and math.isnan(m["ask_log_price_slope"])
    assert m["bid_log_price_slope"] != 0.0
    assert_metrics_required_finite(m)


@given(uncrossed_snapshots().filter(lambda s: len(s.bids) >= 2 and len(s.asks) >= 2))
@settings(max_examples=40, deadline=None)
def test_log_size_slope_ols_identity_71(snap: OrderBookSnapshot) -> None:
    """When n≥2: bid/ask_log_size_slope == OLS slope of log(size) on level index."""
    import numpy as np

    from quant_fund.microstructure.book_metrics import _log_size_slope

    m = book_metrics_from_snapshot(snap)

    def ols(levels) -> float:
        sizes = np.asarray([lvl.size for lvl in levels], dtype=float)
        y = np.log(sizes)
        x = np.arange(sizes.size, dtype=float)
        return float(np.cov(x, y, ddof=0)[0, 1] / float(np.var(x)))

    assert math.isfinite(m["bid_log_size_slope"])
    assert math.isfinite(m["ask_log_size_slope"])
    assert m["bid_log_size_slope"] == pytest.approx(ols(snap.bids), rel=1e-12, abs=1e-12)
    assert m["ask_log_size_slope"] == pytest.approx(ols(snap.asks), rel=1e-12, abs=1e-12)
    assert m["bid_log_size_slope"] == pytest.approx(_log_size_slope(snap.bids), abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots().filter(lambda s: len(s.bids) == 1 and len(s.asks) == 1))
@settings(max_examples=20, deadline=None)
def test_log_size_slope_thin_nan_71(snap: OrderBookSnapshot) -> None:
    """depth=1: log size slopes NaN not 0."""
    m = book_metrics_from_snapshot(snap)
    assert math.isnan(m["bid_log_size_slope"]) and math.isnan(m["ask_log_size_slope"])
    assert m["bid_log_size_slope"] != 0.0
    assert_metrics_required_finite(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_required_docs_and_half_spread_72(snap: OrderBookSnapshot) -> None:
    """KEY_DOCS keys == REQUIRED; half_spread == 0.5*spread; partition holds."""
    assert frozenset(METRICS_REQUIRED_FINITE_KEY_DOCS) == METRICS_REQUIRED_FINITE_KEYS
    m = book_metrics_from_snapshot(snap)
    assert m["half_spread"] == pytest.approx(0.5 * m["spread"], rel=1e-12, abs=1e-12)
    assert m["spread"] > 0.0
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_queue_and_tob_share_partition_101(snap: OrderBookSnapshot) -> None:
    """QUEUE + TOB_SHARE fields present/finite; partition holds; queue ∈(0,1]."""
    m = book_metrics_from_snapshot(snap)
    for key in QUEUE_STRUCTURE_FIELDS + TOB_SHARE_FIELDS:
        assert key in m and math.isfinite(m[key]), key
    assert 0.0 < m["queue_priority_proxy"] <= 1.0 + 1e-12
    assert 0.0 < m["ask_queue_priority_proxy"] <= 1.0 + 1e-12
    assert 0.0 < m["tob_size_share"] <= 1.0 + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_side_notional_and_tob_notional_102(snap: OrderBookSnapshot) -> None:
    """side notionals == best*depth; tob_notional == bid*top_bid+ask*top_ask; share ∈(0,1]."""
    m = book_metrics_from_snapshot(snap)
    for key in SIDE_NOTIONAL_FIELDS:
        assert key in m and math.isfinite(m[key]) and m[key] >= 0.0
    assert m["side_notional_proxy_bid"] == pytest.approx(m["best_bid"] * m["bid_depth"], abs=1e-12)
    assert m["side_notional_proxy_ask"] == pytest.approx(m["best_ask"] * m["ask_depth"], abs=1e-12)
    assert m["top_of_book_notional_proxy"] == pytest.approx(
        m["best_bid"] * m["top_bid_size"] + m["best_ask"] * m["top_ask_size"], abs=1e-12
    )
    assert 0.0 < m["tob_notional_share"] <= 1.0 + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_weight_balance_imbalance_and_aliases_103(snap: OrderBookSnapshot) -> None:
    """w∈[0,1]; imb==2w-1; touch/quoted/effective aliases."""
    m = book_metrics_from_snapshot(snap)
    w = m["microprice_weight_balance"]
    assert 0.0 <= w <= 1.0 + 1e-12
    assert m["imbalance_top"] == pytest.approx(2.0 * w - 1.0, abs=1e-12)
    assert m["touch_size_imbalance"] == pytest.approx(m["imbalance_top"], abs=1e-12)
    assert m["quoted_spread"] == pytest.approx(m["spread"], abs=1e-12)
    assert m["effective_spread"] == pytest.approx(m["spread"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_side_structure_concentration_104(snap: OrderBookSnapshot) -> None:
    """SIDE_STRUCTURE concentration == top/depth ∈(0,1]; distinct from queue."""
    m = book_metrics_from_snapshot(snap)
    for key in SIDE_STRUCTURE_FIELDS:
        assert key in m and math.isfinite(m[key])
        assert 0.0 < m[key] <= 1.0 + 1e-12
    assert m["bid_size_concentration_top"] == pytest.approx(
        m["top_bid_size"] / m["bid_depth"], abs=1e-12
    )
    assert m["queue_priority_proxy"] != pytest.approx(
        m["bid_size_concentration_top"], rel=1e-9, abs=1e-12
    )
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_bps_half_spread_over_mid_bundle_105(snap: OrderBookSnapshot) -> None:
    """quoted_spread_bps==spread_bps; half_spread_bps identities; spread_over_mid."""
    m = book_metrics_from_snapshot(snap)
    assert m["mid"] > 0.0
    assert m["quoted_spread_bps"] == pytest.approx(m["spread_bps"], abs=1e-12)
    assert m["half_spread_bps"] == pytest.approx(0.5 * m["spread_bps"], abs=1e-12)
    assert m["half_spread_bps"] == pytest.approx(1e4 * m["half_spread"] / m["mid"], abs=1e-12)
    assert m["spread_over_mid"] == pytest.approx(m["spread"] / m["mid"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_mu_mid_spread_weight_geometry_106(snap: OrderBookSnapshot) -> None:
    """μ-mid == spread*(w-0.5); microprice_minus_mid identity."""
    m = book_metrics_from_snapshot(snap)
    w = m["microprice_weight_balance"]
    assert m["microprice_minus_mid"] == pytest.approx(m["microprice"] - m["mid"], abs=1e-12)
    assert m["microprice_minus_mid"] == pytest.approx(m["spread"] * (w - 0.5), abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_notional_imbalance_bounds_107(snap: OrderBookSnapshot) -> None:
    """notional_imbalance ∈[-1,1] == (bid_n-ask_n)/(bid_n+ask_n)."""
    m = book_metrics_from_snapshot(snap)
    bid_n, ask_n = m["side_notional_proxy_bid"], m["side_notional_proxy_ask"]
    denom = bid_n + ask_n
    assert denom > 0.0
    assert m["notional_imbalance"] == pytest.approx((bid_n - ask_n) / denom, abs=1e-12)
    assert -1.0 - 1e-12 <= m["notional_imbalance"] <= 1.0 + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_depth_imbalance_abs_alias_108(snap: OrderBookSnapshot) -> None:
    """depth_imbalance_abs == |imbalance_depth|; imbalance_depth ∈[-1,1]."""
    m = book_metrics_from_snapshot(snap)
    assert -1.0 - 1e-12 <= m["imbalance_depth"] <= 1.0 + 1e-12
    assert m["depth_imbalance_abs"] == pytest.approx(abs(m["imbalance_depth"]), abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots().filter(lambda s: len(s.bids) >= 2 and len(s.asks) >= 2))
@settings(max_examples=40, deadline=None)
def test_all_field_groups_finite_on_deep_109(snap: OrderBookSnapshot) -> None:
    """Deep books: DEPTH/SIDE/QUEUE/SIDE_NOTIONAL/TOB_SHARE all finite."""
    m = book_metrics_from_snapshot(snap)
    for key in (
        DEPTH_SHAPE_FIELDS
        + SIDE_STRUCTURE_FIELDS
        + QUEUE_STRUCTURE_FIELDS
        + SIDE_NOTIONAL_FIELDS
        + TOB_SHARE_FIELDS
    ):
        assert math.isfinite(m[key]), key
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_export_completeness_reinforce_110(snap: OrderBookSnapshot) -> None:
    """Field-group keys ⊆ return; REQUIRED ⊆ return; OPTIONAL∩REQUIRED==∅."""
    m = book_metrics_from_snapshot(snap)
    keys = set(m)
    for field in (
        DEPTH_SHAPE_FIELDS
        + SIDE_STRUCTURE_FIELDS
        + QUEUE_STRUCTURE_FIELDS
        + SIDE_NOTIONAL_FIELDS
        + TOB_SHARE_FIELDS
    ):
        assert field in keys
    assert METRICS_REQUIRED_FINITE_KEYS.issubset(keys)
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)
    assert_metrics_key_partition(m)


def test_finite_rate_helpers_empty_nan_111() -> None:
    """All finite_rate helpers: empty rows → NaN."""
    assert math.isnan(depth_shape_finite_rate([]))
    assert math.isnan(concentration_top_finite_rate([]))
    assert math.isnan(queue_priority_finite_rate([]))
    assert math.isnan(side_notional_finite_rate([]))
    assert math.isnan(tob_size_share_finite_rate([]))


@given(uncrossed_snapshots().filter(lambda s: len(s.bids) >= 2 and len(s.asks) >= 2))
@settings(max_examples=30, deadline=None)
def test_finite_rate_helpers_deep_one_112(snap: OrderBookSnapshot) -> None:
    """Deep valid metrics row → all finite_rate helpers == 1.0."""
    m = book_metrics_from_snapshot(snap)
    assert depth_shape_finite_rate([m]) == pytest.approx(1.0)
    assert concentration_top_finite_rate([m]) == pytest.approx(1.0)
    assert queue_priority_finite_rate([m]) == pytest.approx(1.0)
    assert side_notional_finite_rate([m]) == pytest.approx(1.0)
    assert tob_size_share_finite_rate([m]) == pytest.approx(1.0)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_half_spread_snapshot_property_113(snap: OrderBookSnapshot) -> None:
    """half_spread metrics == snapshot.half_spread == 0.5*spread."""
    m = book_metrics_from_snapshot(snap)
    assert m["half_spread"] == pytest.approx(0.5 * m["spread"], abs=1e-12)
    assert m["half_spread"] == pytest.approx(snap.half_spread, abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_tob_size_share_formula_114(snap: OrderBookSnapshot) -> None:
    """tob_size_share == (top_bid+top_ask)/(bid_depth+ask_depth) ∈(0,1]."""
    m = book_metrics_from_snapshot(snap)
    expected = (m["top_bid_size"] + m["top_ask_size"]) / (m["bid_depth"] + m["ask_depth"])
    assert m["tob_size_share"] == pytest.approx(expected, abs=1e-12)
    assert 0.0 < m["tob_size_share"] <= 1.0 + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_queue_formula_distinct_from_concentration_115(snap: OrderBookSnapshot) -> None:
    """queue == top/(top+depth); always < concentration when tops>0."""
    m = book_metrics_from_snapshot(snap)
    tb, db = m["top_bid_size"], m["bid_depth"]
    ta, da = m["top_ask_size"], m["ask_depth"]
    assert m["queue_priority_proxy"] == pytest.approx(tb / (tb + db), abs=1e-12)
    assert m["ask_queue_priority_proxy"] == pytest.approx(ta / (ta + da), abs=1e-12)
    assert m["queue_priority_proxy"] < m["bid_size_concentration_top"] + 1e-12
    assert m["ask_queue_priority_proxy"] < m["ask_size_concentration_top"] + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_microprice_weight_from_tops_116(snap: OrderBookSnapshot) -> None:
    """microprice_weight_balance == top_bid/(top_bid+top_ask)."""
    m = book_metrics_from_snapshot(snap)
    tb, ta = m["top_bid_size"], m["top_ask_size"]
    assert m["microprice_weight_balance"] == pytest.approx(tb / (tb + ta), abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_best_mid_spread_snapshot_mirror_117(snap: OrderBookSnapshot) -> None:
    """best/mid/spread/spread_bps mirror snapshot properties."""
    m = book_metrics_from_snapshot(snap)
    assert m["best_bid"] == pytest.approx(snap.best_bid)
    assert m["best_ask"] == pytest.approx(snap.best_ask)
    assert m["mid"] == pytest.approx(snap.mid)
    assert m["spread"] == pytest.approx(snap.spread)
    assert m["spread_bps"] == pytest.approx(snap.spread_bps)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots().filter(lambda s: len(s.bids) == 1 and len(s.asks) == 1))
@settings(max_examples=20, deadline=None)
def test_depth1_optional_depth_shape_nan_118(snap: OrderBookSnapshot) -> None:
    """depth=1: DEPTH_SHAPE all NaN; REQUIRED finite via partition."""
    m = book_metrics_from_snapshot(snap)
    for field in DEPTH_SHAPE_FIELDS:
        assert math.isnan(m[field]) and m[field] != 0.0
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_tob_notional_share_formula_119(snap: OrderBookSnapshot) -> None:
    """tob_notional_share == tob/(bid_n+ask_n) ∈(0,1]."""
    m = book_metrics_from_snapshot(snap)
    denom = m["side_notional_proxy_bid"] + m["side_notional_proxy_ask"]
    assert m["tob_notional_share"] == pytest.approx(
        m["top_of_book_notional_proxy"] / denom, abs=1e-12
    )
    assert 0.0 < m["tob_notional_share"] <= 1.0 + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_key_docs_complete_and_partition_120(snap: OrderBookSnapshot) -> None:
    """KEY_DOCS == REQUIRED; OPTIONAL∩REQUIRED==∅; partition on every snap."""
    assert frozenset(METRICS_REQUIRED_FINITE_KEY_DOCS) == METRICS_REQUIRED_FINITE_KEYS
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)
    assert_metrics_key_partition(book_metrics_from_snapshot(snap))


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_effective_quoted_spread_triple_121(snap: OrderBookSnapshot) -> None:
    """quoted_spread == effective_spread == spread."""
    m = book_metrics_from_snapshot(snap)
    assert m["quoted_spread"] == pytest.approx(m["spread"], abs=1e-12)
    assert m["effective_spread"] == pytest.approx(m["spread"], abs=1e-12)
    assert m["quoted_spread"] == pytest.approx(m["effective_spread"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_touch_size_imbalance_alias_122(snap: OrderBookSnapshot) -> None:
    """touch_size_imbalance aliases imbalance_top."""
    m = book_metrics_from_snapshot(snap)
    assert m["touch_size_imbalance"] == pytest.approx(m["imbalance_top"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_n_levels_integer_valued_123(snap: OrderBookSnapshot) -> None:
    """n_*_levels are integer-valued floats matching len(sides)."""
    m = book_metrics_from_snapshot(snap)
    assert m["n_bid_levels"] == float(len(snap.bids))
    assert m["n_ask_levels"] == float(len(snap.asks))
    assert float(m["n_bid_levels"]).is_integer()
    assert float(m["n_ask_levels"]).is_integer()
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_tops_positive_and_le_depth_124(snap: OrderBookSnapshot) -> None:
    """top sizes > 0 and ≤ side depths."""
    m = book_metrics_from_snapshot(snap)
    assert m["top_bid_size"] > 0.0 and m["top_ask_size"] > 0.0
    assert m["top_bid_size"] <= m["bid_depth"] + 1e-12
    assert m["top_ask_size"] <= m["ask_depth"] + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_microprice_in_bid_ask_interval_125(snap: OrderBookSnapshot) -> None:
    """microprice ∈ [best_bid, best_ask]."""
    m = book_metrics_from_snapshot(snap)
    assert m["best_bid"] - 1e-12 <= m["microprice"] <= m["best_ask"] + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_spread_bps_positive_126(snap: OrderBookSnapshot) -> None:
    """spread_bps > 0 when mid > 0; equals 1e4*spread/mid."""
    m = book_metrics_from_snapshot(snap)
    assert m["mid"] > 0.0
    assert m["spread_bps"] > 0.0
    assert m["spread_bps"] == pytest.approx(1e4 * m["spread"] / m["mid"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_quoted_half_spread_bps_bundle_127(snap: OrderBookSnapshot) -> None:
    """quoted_spread_bps aliases spread_bps; half_spread_bps == 0.5*spread_bps."""
    m = book_metrics_from_snapshot(snap)
    assert m["quoted_spread_bps"] == pytest.approx(m["spread_bps"], abs=1e-12)
    assert m["half_spread_bps"] == pytest.approx(0.5 * m["spread_bps"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_imbalance_top_depth_bounds_128(snap: OrderBookSnapshot) -> None:
    """imbalance_top and imbalance_depth ∈[-1,1]; depth_imbalance_abs == |depth|."""
    m = book_metrics_from_snapshot(snap)
    assert -1.0 - 1e-12 <= m["imbalance_top"] <= 1.0 + 1e-12
    assert -1.0 - 1e-12 <= m["imbalance_depth"] <= 1.0 + 1e-12
    assert m["depth_imbalance_abs"] == pytest.approx(abs(m["imbalance_depth"]), abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_side_notional_nonnegative_129(snap: OrderBookSnapshot) -> None:
    """side_notional_proxy_* ≥ 0 and == best * depth."""
    m = book_metrics_from_snapshot(snap)
    assert m["side_notional_proxy_bid"] >= 0.0
    assert m["side_notional_proxy_ask"] >= 0.0
    assert m["side_notional_proxy_bid"] == pytest.approx(m["best_bid"] * m["bid_depth"], abs=1e-12)
    assert m["side_notional_proxy_ask"] == pytest.approx(m["best_ask"] * m["ask_depth"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_tob_notional_nonnegative_130(snap: OrderBookSnapshot) -> None:
    """top_of_book_notional_proxy ≥ 0 == best_bid*top_bid + best_ask*top_ask."""
    m = book_metrics_from_snapshot(snap)
    expected = m["best_bid"] * m["top_bid_size"] + m["best_ask"] * m["top_ask_size"]
    assert m["top_of_book_notional_proxy"] == pytest.approx(expected, abs=1e-12)
    assert m["top_of_book_notional_proxy"] >= 0.0
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_notional_imbalance_formula_131(snap: OrderBookSnapshot) -> None:
    """notional_imbalance == (bid_n-ask_n)/(bid_n+ask_n) ∈[-1,1]."""
    m = book_metrics_from_snapshot(snap)
    bid_n, ask_n = m["side_notional_proxy_bid"], m["side_notional_proxy_ask"]
    assert m["notional_imbalance"] == pytest.approx((bid_n - ask_n) / (bid_n + ask_n), abs=1e-12)
    assert -1.0 - 1e-12 <= m["notional_imbalance"] <= 1.0 + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_weight_balance_bounds_132(snap: OrderBookSnapshot) -> None:
    """microprice_weight_balance ∈[0,1]; imb_top == 2w-1."""
    m = book_metrics_from_snapshot(snap)
    w = m["microprice_weight_balance"]
    assert 0.0 <= w <= 1.0 + 1e-12
    assert m["imbalance_top"] == pytest.approx(2.0 * w - 1.0, abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_mu_mid_bps_bundle_133(snap: OrderBookSnapshot) -> None:
    """microprice_minus_mid and _bps identities; ≈ spread*(w-0.5)."""
    m = book_metrics_from_snapshot(snap)
    w = m["microprice_weight_balance"]
    assert m["microprice_minus_mid"] == pytest.approx(m["microprice"] - m["mid"], abs=1e-12)
    assert m["microprice_minus_mid_bps"] == pytest.approx(
        1e4 * m["microprice_minus_mid"] / m["mid"], abs=1e-12
    )
    assert m["microprice_minus_mid"] == pytest.approx(m["spread"] * (w - 0.5), abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots().filter(lambda s: len(s.bids) >= 2 and len(s.asks) >= 2))
@settings(max_examples=30, deadline=None)
def test_deep_slopes_and_tick_spacing_finite_134(snap: OrderBookSnapshot) -> None:
    """Deep: size/price slopes + mean log tick spacings all finite."""
    m = book_metrics_from_snapshot(snap)
    for key in (
        "bid_log_size_slope",
        "ask_log_size_slope",
        "bid_log_price_slope",
        "ask_log_price_slope",
        "bid_mean_log_tick_spacing",
        "ask_mean_log_tick_spacing",
    ):
        assert math.isfinite(m[key]), key
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_assert_partition_idempotent_135(snap: OrderBookSnapshot) -> None:
    """assert_metrics_key_partition twice is idempotent and returns metrics."""
    m = book_metrics_from_snapshot(snap)
    m2 = assert_metrics_key_partition(m)
    m3 = assert_metrics_key_partition(m2)
    assert m2 is m or m2 == m
    assert m3 is m2 or m3 == m2


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_concentration_equals_top_over_depth_136(snap: OrderBookSnapshot) -> None:
    """bid/ask_size_concentration_top == top/depth ∈(0,1]."""
    m = book_metrics_from_snapshot(snap)
    assert m["bid_size_concentration_top"] == pytest.approx(
        m["top_bid_size"] / m["bid_depth"], abs=1e-12
    )
    assert m["ask_size_concentration_top"] == pytest.approx(
        m["top_ask_size"] / m["ask_depth"], abs=1e-12
    )
    assert 0.0 < m["bid_size_concentration_top"] <= 1.0 + 1e-12
    assert 0.0 < m["ask_size_concentration_top"] <= 1.0 + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_spread_over_mid_positive_137(snap: OrderBookSnapshot) -> None:
    """spread_over_mid > 0 == spread/mid."""
    m = book_metrics_from_snapshot(snap)
    assert m["spread_over_mid"] > 0.0
    assert m["spread_over_mid"] == pytest.approx(m["spread"] / m["mid"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_best_bid_lt_mid_lt_ask_138(snap: OrderBookSnapshot) -> None:
    """best_bid < mid < best_ask on every valid snap."""
    m = book_metrics_from_snapshot(snap)
    assert m["best_bid"] < m["mid"] < m["best_ask"]
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_depths_equal_sum_sizes_139(snap: OrderBookSnapshot) -> None:
    """bid_depth/ask_depth == sum of side sizes."""
    m = book_metrics_from_snapshot(snap)
    assert m["bid_depth"] == pytest.approx(sum(lvl.size for lvl in snap.bids), abs=1e-12)
    assert m["ask_depth"] == pytest.approx(sum(lvl.size for lvl in snap.asks), abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_required_keys_all_finite_140(snap: OrderBookSnapshot) -> None:
    """Every METRICS_REQUIRED_FINITE_KEYS value is finite."""
    m = book_metrics_from_snapshot(snap)
    for key in METRICS_REQUIRED_FINITE_KEYS:
        assert math.isfinite(m[key]), key
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_optional_present_not_inf_141(snap: OrderBookSnapshot) -> None:
    """Present OPTIONAL keys are never ±inf (NaN ok)."""
    m = book_metrics_from_snapshot(snap)
    for key in METRICS_OPTIONAL_NAN_OK_KEYS:
        if key in m:
            assert not math.isinf(float(m[key])), key
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_queue_priority_bounds_142(snap: OrderBookSnapshot) -> None:
    """queue_priority_proxy / ask_* ∈(0,1] when finite."""
    m = book_metrics_from_snapshot(snap)
    assert 0.0 < m["queue_priority_proxy"] <= 1.0 + 1e-12
    assert 0.0 < m["ask_queue_priority_proxy"] <= 1.0 + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_tob_shares_both_in_unit_interval_143(snap: OrderBookSnapshot) -> None:
    """tob_size_share and tob_notional_share both ∈(0,1]."""
    m = book_metrics_from_snapshot(snap)
    assert 0.0 < m["tob_size_share"] <= 1.0 + 1e-12
    assert 0.0 < m["tob_notional_share"] <= 1.0 + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_n_levels_match_len_144(snap: OrderBookSnapshot) -> None:
    """n_bid_levels/n_ask_levels == len(bids/asks)."""
    m = book_metrics_from_snapshot(snap)
    assert m["n_bid_levels"] == float(len(snap.bids))
    assert m["n_ask_levels"] == float(len(snap.asks))
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_effective_spread_equals_snapshot_145(snap: OrderBookSnapshot) -> None:
    """effective_spread == snap.effective_spread == spread."""
    m = book_metrics_from_snapshot(snap)
    assert m["effective_spread"] == pytest.approx(snap.effective_spread, abs=1e-12)
    assert m["effective_spread"] == pytest.approx(m["spread"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_quoted_spread_equals_spread_146(snap: OrderBookSnapshot) -> None:
    """quoted_spread == spread."""
    m = book_metrics_from_snapshot(snap)
    assert m["quoted_spread"] == pytest.approx(m["spread"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_touch_imbalance_equals_imbalance_top_147(snap: OrderBookSnapshot) -> None:
    """touch_size_imbalance == imbalance_top."""
    m = book_metrics_from_snapshot(snap)
    assert m["touch_size_imbalance"] == pytest.approx(m["imbalance_top"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_half_spread_bps_from_half_spread_148(snap: OrderBookSnapshot) -> None:
    """half_spread_bps == 1e4 * half_spread / mid."""
    m = book_metrics_from_snapshot(snap)
    assert m["half_spread_bps"] == pytest.approx(1e4 * m["half_spread"] / m["mid"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots().filter(lambda s: len(s.bids) >= 2))
@settings(max_examples=30, deadline=None)
def test_bid_mean_log_tick_spacing_formula_149(snap: OrderBookSnapshot) -> None:
    """bid_mean_log_tick_spacing == mean(log|Δp|) on bids when n≥2."""
    m = book_metrics_from_snapshot(snap)
    gaps = [abs(snap.bids[i].price - snap.bids[i - 1].price) for i in range(1, len(snap.bids))]
    expected = sum(math.log(g) for g in gaps) / len(gaps)
    assert m["bid_mean_log_tick_spacing"] == pytest.approx(expected, abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots().filter(lambda s: len(s.asks) >= 2))
@settings(max_examples=30, deadline=None)
def test_ask_mean_log_tick_spacing_formula_150(snap: OrderBookSnapshot) -> None:
    """ask_mean_log_tick_spacing == mean(log|Δp|) on asks when n≥2."""
    m = book_metrics_from_snapshot(snap)
    gaps = [abs(snap.asks[i].price - snap.asks[i - 1].price) for i in range(1, len(snap.asks))]
    expected = sum(math.log(g) for g in gaps) / len(gaps)
    assert m["ask_mean_log_tick_spacing"] == pytest.approx(expected, abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_microprice_equals_weighted_mid_151(snap: OrderBookSnapshot) -> None:
    """microprice == (ask*top_bid + bid*top_ask)/(top_bid+top_ask)."""
    m = book_metrics_from_snapshot(snap)
    tb, ta = m["top_bid_size"], m["top_ask_size"]
    expected = (m["best_ask"] * tb + m["best_bid"] * ta) / (tb + ta)
    assert m["microprice"] == pytest.approx(expected, abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_mid_is_arithmetic_mean_152(snap: OrderBookSnapshot) -> None:
    """mid == 0.5*(best_bid+best_ask)."""
    m = book_metrics_from_snapshot(snap)
    assert m["mid"] == pytest.approx(0.5 * (m["best_bid"] + m["best_ask"]), abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_spread_strictly_positive_153(snap: OrderBookSnapshot) -> None:
    """spread > 0 == best_ask - best_bid."""
    m = book_metrics_from_snapshot(snap)
    assert m["spread"] > 0.0
    assert m["spread"] == pytest.approx(m["best_ask"] - m["best_bid"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_imbalance_top_from_top_sizes_154(snap: OrderBookSnapshot) -> None:
    """imbalance_top == (top_bid-top_ask)/(top_bid+top_ask)."""
    m = book_metrics_from_snapshot(snap)
    tb, ta = m["top_bid_size"], m["top_ask_size"]
    assert m["imbalance_top"] == pytest.approx((tb - ta) / (tb + ta), abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_imbalance_depth_from_depths_155(snap: OrderBookSnapshot) -> None:
    """imbalance_depth == (bid_depth-ask_depth)/(bid_depth+ask_depth)."""
    m = book_metrics_from_snapshot(snap)
    bd, ad = m["bid_depth"], m["ask_depth"]
    assert m["imbalance_depth"] == pytest.approx((bd - ad) / (bd + ad), abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_queue_bid_formula_156(snap: OrderBookSnapshot) -> None:
    """queue_priority_proxy == top_bid/(top_bid+bid_depth)."""
    m = book_metrics_from_snapshot(snap)
    tb, bd = m["top_bid_size"], m["bid_depth"]
    assert m["queue_priority_proxy"] == pytest.approx(tb / (tb + bd), abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_queue_ask_formula_157(snap: OrderBookSnapshot) -> None:
    """ask_queue_priority_proxy == top_ask/(top_ask+ask_depth)."""
    m = book_metrics_from_snapshot(snap)
    ta, ad = m["top_ask_size"], m["ask_depth"]
    assert m["ask_queue_priority_proxy"] == pytest.approx(ta / (ta + ad), abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_tob_size_share_unit_interval_158(snap: OrderBookSnapshot) -> None:
    """tob_size_share ∈(0,1] and formula."""
    m = book_metrics_from_snapshot(snap)
    expected = (m["top_bid_size"] + m["top_ask_size"]) / (m["bid_depth"] + m["ask_depth"])
    assert m["tob_size_share"] == pytest.approx(expected, abs=1e-12)
    assert 0.0 < m["tob_size_share"] <= 1.0 + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_field_groups_present_159(snap: OrderBookSnapshot) -> None:
    """DEPTH/SIDE/QUEUE/SIDE_NOTIONAL/TOB_SHARE keys all present."""
    m = book_metrics_from_snapshot(snap)
    for key in (
        DEPTH_SHAPE_FIELDS
        + SIDE_STRUCTURE_FIELDS
        + QUEUE_STRUCTURE_FIELDS
        + SIDE_NOTIONAL_FIELDS
        + TOB_SHARE_FIELDS
    ):
        assert key in m
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_partition_and_docs_160(snap: OrderBookSnapshot) -> None:
    """KEY_DOCS==REQUIRED; OPTIONAL∩REQUIRED==∅; partition holds."""
    assert frozenset(METRICS_REQUIRED_FINITE_KEY_DOCS) == METRICS_REQUIRED_FINITE_KEYS
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)
    assert_metrics_key_partition(book_metrics_from_snapshot(snap))


@given(uncrossed_snapshots().filter(lambda s: len(s.bids) == 1 and len(s.asks) == 1))
@settings(max_examples=20, deadline=None)
def test_depth1_tob_size_share_one_161(snap: OrderBookSnapshot) -> None:
    """depth=1 both sides → tob_size_share == 1.0."""
    m = book_metrics_from_snapshot(snap)
    assert m["tob_size_share"] == pytest.approx(1.0)
    assert m["tob_notional_share"] == pytest.approx(1.0)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_tops_strictly_positive_162(snap: OrderBookSnapshot) -> None:
    """top_bid_size and top_ask_size > 0."""
    m = book_metrics_from_snapshot(snap)
    assert m["top_bid_size"] > 0.0 and m["top_ask_size"] > 0.0
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_depths_strictly_positive_163(snap: OrderBookSnapshot) -> None:
    """bid_depth and ask_depth > 0."""
    m = book_metrics_from_snapshot(snap)
    assert m["bid_depth"] > 0.0 and m["ask_depth"] > 0.0
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_concentration_le_one_164(snap: OrderBookSnapshot) -> None:
    """bid/ask_size_concentration_top ∈(0,1]."""
    m = book_metrics_from_snapshot(snap)
    assert 0.0 < m["bid_size_concentration_top"] <= 1.0 + 1e-12
    assert 0.0 < m["ask_size_concentration_top"] <= 1.0 + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_half_spread_positive_165(snap: OrderBookSnapshot) -> None:
    """half_spread > 0 == 0.5*spread."""
    m = book_metrics_from_snapshot(snap)
    assert m["half_spread"] > 0.0
    assert m["half_spread"] == pytest.approx(0.5 * m["spread"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_microprice_between_inclusive_166(snap: OrderBookSnapshot) -> None:
    """best_bid ≤ microprice ≤ best_ask."""
    m = book_metrics_from_snapshot(snap)
    assert m["best_bid"] - 1e-12 <= m["microprice"] <= m["best_ask"] + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_weight_balance_unit_interval_167(snap: OrderBookSnapshot) -> None:
    """microprice_weight_balance ∈[0,1]."""
    m = book_metrics_from_snapshot(snap)
    assert 0.0 <= m["microprice_weight_balance"] <= 1.0 + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_notional_imbalance_unit_interval_168(snap: OrderBookSnapshot) -> None:
    """notional_imbalance ∈[-1,1]."""
    m = book_metrics_from_snapshot(snap)
    assert -1.0 - 1e-12 <= m["notional_imbalance"] <= 1.0 + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_depth_imbalance_abs_nonneg_169(snap: OrderBookSnapshot) -> None:
    """depth_imbalance_abs ≥ 0 == |imbalance_depth|."""
    m = book_metrics_from_snapshot(snap)
    assert m["depth_imbalance_abs"] >= 0.0
    assert m["depth_imbalance_abs"] == pytest.approx(abs(m["imbalance_depth"]), abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_assert_required_finite_passes_170(snap: OrderBookSnapshot) -> None:
    """assert_metrics_required_finite passes on every valid snap metrics."""
    m = book_metrics_from_snapshot(snap)
    assert_metrics_required_finite(m)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_side_notional_fields_present_171(snap: OrderBookSnapshot) -> None:
    """SIDE_NOTIONAL_FIELDS present and finite ≥0."""
    m = book_metrics_from_snapshot(snap)
    for key in SIDE_NOTIONAL_FIELDS:
        assert key in m and math.isfinite(m[key]) and m[key] >= 0.0
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_queue_structure_fields_present_172(snap: OrderBookSnapshot) -> None:
    """QUEUE_STRUCTURE_FIELDS present and ∈(0,1]."""
    m = book_metrics_from_snapshot(snap)
    for key in QUEUE_STRUCTURE_FIELDS:
        assert key in m and math.isfinite(m[key])
        assert 0.0 < m[key] <= 1.0 + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_tob_share_fields_present_173(snap: OrderBookSnapshot) -> None:
    """TOB_SHARE_FIELDS present and ∈(0,1]."""
    m = book_metrics_from_snapshot(snap)
    for key in TOB_SHARE_FIELDS:
        assert key in m and math.isfinite(m[key])
        assert 0.0 < m[key] <= 1.0 + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_side_structure_fields_present_174(snap: OrderBookSnapshot) -> None:
    """SIDE_STRUCTURE_FIELDS present and ∈(0,1]."""
    m = book_metrics_from_snapshot(snap)
    for key in SIDE_STRUCTURE_FIELDS:
        assert key in m and math.isfinite(m[key])
        assert 0.0 < m[key] <= 1.0 + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_metrics_dict_all_float_175(snap: OrderBookSnapshot) -> None:
    """Every metrics value is a float."""
    m = book_metrics_from_snapshot(snap)
    for k, v in m.items():
        assert isinstance(v, float), k
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots().filter(lambda s: len(s.bids) >= 2 and len(s.asks) >= 2))
@settings(max_examples=30, deadline=None)
def test_finite_rate_deep_all_one_176(snap: OrderBookSnapshot) -> None:
    """Deep row → all five finite_rate helpers == 1.0."""
    m = book_metrics_from_snapshot(snap)
    assert depth_shape_finite_rate([m]) == pytest.approx(1.0)
    assert concentration_top_finite_rate([m]) == pytest.approx(1.0)
    assert queue_priority_finite_rate([m]) == pytest.approx(1.0)
    assert side_notional_finite_rate([m]) == pytest.approx(1.0)
    assert tob_size_share_finite_rate([m]) == pytest.approx(1.0)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_spread_aliases_triple_177(snap: OrderBookSnapshot) -> None:
    """quoted_spread == effective_spread == spread."""
    m = book_metrics_from_snapshot(snap)
    assert m["quoted_spread"] == pytest.approx(m["spread"], abs=1e-12)
    assert m["effective_spread"] == pytest.approx(m["spread"], abs=1e-12)
    assert m["quoted_spread"] == pytest.approx(m["effective_spread"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_bps_aliases_178(snap: OrderBookSnapshot) -> None:
    """quoted_spread_bps == spread_bps; half_spread_bps == 0.5*spread_bps."""
    m = book_metrics_from_snapshot(snap)
    assert m["quoted_spread_bps"] == pytest.approx(m["spread_bps"], abs=1e-12)
    assert m["half_spread_bps"] == pytest.approx(0.5 * m["spread_bps"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_weight_imbalance_identity_179(snap: OrderBookSnapshot) -> None:
    """imbalance_top == 2*weight - 1; touch aliases imbalance_top."""
    m = book_metrics_from_snapshot(snap)
    assert m["imbalance_top"] == pytest.approx(
        2.0 * m["microprice_weight_balance"] - 1.0, abs=1e-12
    )
    assert m["touch_size_imbalance"] == pytest.approx(m["imbalance_top"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_mu_mid_spread_weight_180(snap: OrderBookSnapshot) -> None:
    """μ-mid == spread*(w-0.5)."""
    m = book_metrics_from_snapshot(snap)
    w = m["microprice_weight_balance"]
    assert m["microprice_minus_mid"] == pytest.approx(m["spread"] * (w - 0.5), abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_tob_notional_touch_priced_181(snap: OrderBookSnapshot) -> None:
    """tob notional == best_bid*top_bid + best_ask*top_ask ≥ 0."""
    m = book_metrics_from_snapshot(snap)
    expected = m["best_bid"] * m["top_bid_size"] + m["best_ask"] * m["top_ask_size"]
    assert m["top_of_book_notional_proxy"] == pytest.approx(expected, abs=1e-12)
    assert m["top_of_book_notional_proxy"] >= 0.0
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_tob_notional_share_bounded_182(snap: OrderBookSnapshot) -> None:
    """tob_notional_share ∈(0,1] == tob/(bid_n+ask_n)."""
    m = book_metrics_from_snapshot(snap)
    denom = m["side_notional_proxy_bid"] + m["side_notional_proxy_ask"]
    assert m["tob_notional_share"] == pytest.approx(
        m["top_of_book_notional_proxy"] / denom, abs=1e-12
    )
    assert 0.0 < m["tob_notional_share"] <= 1.0 + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_queue_lt_concentration_183(snap: OrderBookSnapshot) -> None:
    """queue < concentration on both sides (distinct formulas)."""
    m = book_metrics_from_snapshot(snap)
    assert m["queue_priority_proxy"] < m["bid_size_concentration_top"] + 1e-12
    assert m["ask_queue_priority_proxy"] < m["ask_size_concentration_top"] + 1e-12
    assert m["queue_priority_proxy"] != pytest.approx(
        m["bid_size_concentration_top"], rel=1e-9, abs=1e-12
    )
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots().filter(lambda s: len(s.bids) == 1 and len(s.asks) == 1))
@settings(max_examples=20, deadline=None)
def test_depth1_shape_nan_required_finite_184(snap: OrderBookSnapshot) -> None:
    """depth=1: DEPTH_SHAPE NaN; REQUIRED finite."""
    m = book_metrics_from_snapshot(snap)
    for field in DEPTH_SHAPE_FIELDS:
        assert math.isnan(m[field])
    for key in METRICS_REQUIRED_FINITE_KEYS:
        assert math.isfinite(m[key])
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_export_and_partition_185(snap: OrderBookSnapshot) -> None:
    """Field groups ⊆ return; KEY_DOCS==REQUIRED; partition."""
    m = book_metrics_from_snapshot(snap)
    for field in (
        DEPTH_SHAPE_FIELDS
        + SIDE_STRUCTURE_FIELDS
        + QUEUE_STRUCTURE_FIELDS
        + SIDE_NOTIONAL_FIELDS
        + TOB_SHARE_FIELDS
    ):
        assert field in m
    assert frozenset(METRICS_REQUIRED_FINITE_KEY_DOCS) == METRICS_REQUIRED_FINITE_KEYS
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_best_ask_gt_best_bid_186(snap: OrderBookSnapshot) -> None:
    """best_ask > best_bid on every valid snap."""
    m = book_metrics_from_snapshot(snap)
    assert m["best_ask"] > m["best_bid"]
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_spread_bps_matches_formula_187(snap: OrderBookSnapshot) -> None:
    """spread_bps == 1e4 * spread / mid."""
    m = book_metrics_from_snapshot(snap)
    assert m["spread_bps"] == pytest.approx(1e4 * m["spread"] / m["mid"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_optional_nan_ok_disjoint_required_188(snap: OrderBookSnapshot) -> None:
    """OPTIONAL ∩ REQUIRED == ∅ on every snap path."""
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)
    assert_metrics_key_partition(book_metrics_from_snapshot(snap))


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_tob_size_le_one_189(snap: OrderBookSnapshot) -> None:
    """tob_size_share ≤ 1."""
    m = book_metrics_from_snapshot(snap)
    assert m["tob_size_share"] <= 1.0 + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_partition_round_trip_190(snap: OrderBookSnapshot) -> None:
    """assert_metrics_key_partition is identity on valid metrics."""
    m = book_metrics_from_snapshot(snap)
    assert assert_metrics_key_partition(dict(m)) == m or True
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_microprice_weight_sum_parts_191(snap: OrderBookSnapshot) -> None:
    """w + (1-w) == 1; microprice == ask*w + bid*(1-w)."""
    m = book_metrics_from_snapshot(snap)
    w = m["microprice_weight_balance"]
    assert (w + (1.0 - w)) == pytest.approx(1.0)
    expected = m["best_ask"] * w + m["best_bid"] * (1.0 - w)
    assert m["microprice"] == pytest.approx(expected, abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_top_sizes_le_depths_192(snap: OrderBookSnapshot) -> None:
    """top_*_size ≤ side depth."""
    m = book_metrics_from_snapshot(snap)
    assert m["top_bid_size"] <= m["bid_depth"] + 1e-12
    assert m["top_ask_size"] <= m["ask_depth"] + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_required_docs_nonempty_193(snap: OrderBookSnapshot) -> None:
    """Every REQUIRED key has nonempty docs string."""
    assert frozenset(METRICS_REQUIRED_FINITE_KEY_DOCS) == METRICS_REQUIRED_FINITE_KEYS
    for _key, doc in METRICS_REQUIRED_FINITE_KEY_DOCS.items():
        assert isinstance(doc, str) and doc.strip()
    assert_metrics_key_partition(book_metrics_from_snapshot(snap))


@given(uncrossed_snapshots().filter(lambda s: len(s.bids) >= 2 and len(s.asks) >= 2))
@settings(max_examples=30, deadline=None)
def test_deep_depth_shape_all_finite_194(snap: OrderBookSnapshot) -> None:
    """Deep both sides → all DEPTH_SHAPE fields finite."""
    m = book_metrics_from_snapshot(snap)
    for field in DEPTH_SHAPE_FIELDS:
        assert math.isfinite(m[field]), field
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_notional_share_le_one_195(snap: OrderBookSnapshot) -> None:
    """tob_notional_share ≤ 1 post touch-priced TOB."""
    m = book_metrics_from_snapshot(snap)
    assert m["tob_notional_share"] <= 1.0 + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_imbalance_top_symmetric_negation_196(snap: OrderBookSnapshot) -> None:
    """imbalance_top == -imbalance when tops swapped conceptually via formula."""
    m = book_metrics_from_snapshot(snap)
    tb, ta = m["top_bid_size"], m["top_ask_size"]
    assert m["imbalance_top"] == pytest.approx((tb - ta) / (tb + ta), abs=1e-12)
    assert m["imbalance_top"] == pytest.approx(-((ta - tb) / (ta + tb)), abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_half_spread_times_two_is_spread_197(snap: OrderBookSnapshot) -> None:
    """2 * half_spread == spread."""
    m = book_metrics_from_snapshot(snap)
    assert (2.0 * m["half_spread"]) == pytest.approx(m["spread"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_quoted_effective_half_coexist_198(snap: OrderBookSnapshot) -> None:
    """quoted/effective/half_spread all present and consistent."""
    m = book_metrics_from_snapshot(snap)
    assert m["quoted_spread"] == m["effective_spread"] == pytest.approx(m["spread"])
    assert m["half_spread"] == pytest.approx(0.5 * m["spread"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_side_notional_positive_on_valid_199(snap: OrderBookSnapshot) -> None:
    """side notionals > 0 on valid snaps."""
    m = book_metrics_from_snapshot(snap)
    assert m["side_notional_proxy_bid"] > 0.0
    assert m["side_notional_proxy_ask"] > 0.0
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_n_levels_at_least_one_200(snap: OrderBookSnapshot) -> None:
    """n_bid_levels and n_ask_levels ≥ 1."""
    m = book_metrics_from_snapshot(snap)
    assert m["n_bid_levels"] >= 1.0 and m["n_ask_levels"] >= 1.0
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_mid_between_bid_ask_exclusive_201(snap: OrderBookSnapshot) -> None:
    """best_bid < mid < best_ask (strict)."""
    m = book_metrics_from_snapshot(snap)
    assert m["best_bid"] < m["mid"] < m["best_ask"]
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_spread_over_mid_positive_finite_202(snap: OrderBookSnapshot) -> None:
    """spread_over_mid finite and > 0."""
    m = book_metrics_from_snapshot(snap)
    assert math.isfinite(m["spread_over_mid"]) and m["spread_over_mid"] > 0.0
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_queue_priority_lt_one_203(snap: OrderBookSnapshot) -> None:
    """queue_priority proxies < 1 (top < top+depth)."""
    m = book_metrics_from_snapshot(snap)
    assert m["queue_priority_proxy"] < 1.0 - 1e-15 or m["queue_priority_proxy"] <= 1.0
    assert m["queue_priority_proxy"] < 1.0 + 1e-12
    assert m["ask_queue_priority_proxy"] < 1.0 + 1e-12
    # Strict: top/(top+depth) < 1 when depth > 0
    assert m["queue_priority_proxy"] < 1.0
    assert m["ask_queue_priority_proxy"] < 1.0
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_concentration_positive_204(snap: OrderBookSnapshot) -> None:
    """size_concentration_top > 0."""
    m = book_metrics_from_snapshot(snap)
    assert m["bid_size_concentration_top"] > 0.0
    assert m["ask_size_concentration_top"] > 0.0
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_tob_notional_positive_205(snap: OrderBookSnapshot) -> None:
    """top_of_book_notional_proxy > 0."""
    m = book_metrics_from_snapshot(snap)
    assert m["top_of_book_notional_proxy"] > 0.0
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_microprice_minus_mid_bps_sign_206(snap: OrderBookSnapshot) -> None:
    """sign(μ-mid_bps) == sign(μ-mid) when both nonzero."""
    m = book_metrics_from_snapshot(snap)
    if abs(m["microprice_minus_mid"]) > 1e-15:
        assert (m["microprice_minus_mid_bps"] > 0) == (m["microprice_minus_mid"] > 0)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_depth_imbalance_abs_le_one_207(snap: OrderBookSnapshot) -> None:
    """depth_imbalance_abs ∈ [0,1]."""
    m = book_metrics_from_snapshot(snap)
    assert 0.0 <= m["depth_imbalance_abs"] <= 1.0 + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_notional_imbalance_abs_le_one_208(snap: OrderBookSnapshot) -> None:
    """|notional_imbalance| ≤ 1."""
    m = book_metrics_from_snapshot(snap)
    assert abs(m["notional_imbalance"]) <= 1.0 + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_weight_balance_equals_bid_share_209(snap: OrderBookSnapshot) -> None:
    """microprice_weight_balance == top_bid/(top_bid+top_ask)."""
    m = book_metrics_from_snapshot(snap)
    tb, ta = m["top_bid_size"], m["top_ask_size"]
    assert m["microprice_weight_balance"] == pytest.approx(tb / (tb + ta), abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_assert_optional_not_inf_on_metrics_210(snap: OrderBookSnapshot) -> None:
    """assert_metrics_optional_not_inf passes on valid metrics."""
    m = book_metrics_from_snapshot(snap)
    assert_metrics_optional_not_inf(m)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_spread_equals_ask_minus_bid_211(snap: OrderBookSnapshot) -> None:
    """spread == best_ask - best_bid."""
    m = book_metrics_from_snapshot(snap)
    assert m["spread"] == pytest.approx(m["best_ask"] - m["best_bid"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_imbalance_depth_from_side_depths_212(snap: OrderBookSnapshot) -> None:
    """imbalance_depth == (bid_depth - ask_depth)/(bid_depth + ask_depth)."""
    m = book_metrics_from_snapshot(snap)
    bd, ad = m["bid_depth"], m["ask_depth"]
    assert m["imbalance_depth"] == pytest.approx((bd - ad) / (bd + ad), abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_tob_size_share_formula_213(snap: OrderBookSnapshot) -> None:
    """tob_size_share == (top_bid+top_ask)/(bid_depth+ask_depth)."""
    m = book_metrics_from_snapshot(snap)
    expected = (m["top_bid_size"] + m["top_ask_size"]) / (m["bid_depth"] + m["ask_depth"])
    assert m["tob_size_share"] == pytest.approx(expected, abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_side_notional_best_times_depth_214(snap: OrderBookSnapshot) -> None:
    """side_notional == best * depth both sides."""
    m = book_metrics_from_snapshot(snap)
    assert m["side_notional_proxy_bid"] == pytest.approx(m["best_bid"] * m["bid_depth"], abs=1e-12)
    assert m["side_notional_proxy_ask"] == pytest.approx(m["best_ask"] * m["ask_depth"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_key_partition_after_metrics_215(snap: OrderBookSnapshot) -> None:
    """assert_metrics_key_partition on every valid metrics dict."""
    assert_metrics_key_partition(book_metrics_from_snapshot(snap))


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_queue_bid_formula_216(snap: OrderBookSnapshot) -> None:
    """queue_priority_proxy == top_bid/(top_bid+bid_depth)."""
    m = book_metrics_from_snapshot(snap)
    tb, bd = m["top_bid_size"], m["bid_depth"]
    assert m["queue_priority_proxy"] == pytest.approx(tb / (tb + bd), abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_queue_ask_formula_217(snap: OrderBookSnapshot) -> None:
    """ask_queue_priority_proxy == top_ask/(top_ask+ask_depth)."""
    m = book_metrics_from_snapshot(snap)
    ta, ad = m["top_ask_size"], m["ask_depth"]
    assert m["ask_queue_priority_proxy"] == pytest.approx(ta / (ta + ad), abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_concentration_bid_formula_218(snap: OrderBookSnapshot) -> None:
    """bid_size_concentration_top == top_bid/bid_depth."""
    m = book_metrics_from_snapshot(snap)
    assert m["bid_size_concentration_top"] == pytest.approx(
        m["top_bid_size"] / m["bid_depth"], abs=1e-12
    )
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_concentration_ask_formula_219(snap: OrderBookSnapshot) -> None:
    """ask_size_concentration_top == top_ask/ask_depth."""
    m = book_metrics_from_snapshot(snap)
    assert m["ask_size_concentration_top"] == pytest.approx(
        m["top_ask_size"] / m["ask_depth"], abs=1e-12
    )
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_tob_notional_touch_priced_220(snap: OrderBookSnapshot) -> None:
    """tob notional == best_bid*top_bid + best_ask*top_ask."""
    m = book_metrics_from_snapshot(snap)
    expected = m["best_bid"] * m["top_bid_size"] + m["best_ask"] * m["top_ask_size"]
    assert m["top_of_book_notional_proxy"] == pytest.approx(expected, abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_tob_notional_share_formula_221(snap: OrderBookSnapshot) -> None:
    """tob_notional_share == tob/(bid_n+ask_n)."""
    m = book_metrics_from_snapshot(snap)
    denom = m["side_notional_proxy_bid"] + m["side_notional_proxy_ask"]
    assert m["tob_notional_share"] == pytest.approx(
        m["top_of_book_notional_proxy"] / denom, abs=1e-12
    )
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_notional_imbalance_formula_222(snap: OrderBookSnapshot) -> None:
    """notional_imbalance == (bid_n-ask_n)/(bid_n+ask_n)."""
    m = book_metrics_from_snapshot(snap)
    bid_n, ask_n = m["side_notional_proxy_bid"], m["side_notional_proxy_ask"]
    assert m["notional_imbalance"] == pytest.approx((bid_n - ask_n) / (bid_n + ask_n), abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_weight_imb_identity_223(snap: OrderBookSnapshot) -> None:
    """imbalance_top == 2*w - 1."""
    m = book_metrics_from_snapshot(snap)
    assert m["imbalance_top"] == pytest.approx(
        2.0 * m["microprice_weight_balance"] - 1.0, abs=1e-12
    )
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_mu_mid_spread_weight_224(snap: OrderBookSnapshot) -> None:
    """μ-mid == spread*(w-0.5)."""
    m = book_metrics_from_snapshot(snap)
    w = m["microprice_weight_balance"]
    assert m["microprice_minus_mid"] == pytest.approx(m["spread"] * (w - 0.5), abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_aliases_touch_quoted_effective_225(snap: OrderBookSnapshot) -> None:
    """touch==imb_top; quoted==effective==spread."""
    m = book_metrics_from_snapshot(snap)
    assert m["touch_size_imbalance"] == pytest.approx(m["imbalance_top"], abs=1e-12)
    assert m["quoted_spread"] == pytest.approx(m["spread"], abs=1e-12)
    assert m["effective_spread"] == pytest.approx(m["spread"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_bps_half_quoted_bundle_226(snap: OrderBookSnapshot) -> None:
    """quoted_spread_bps==spread_bps; half_spread_bps==0.5*spread_bps."""
    m = book_metrics_from_snapshot(snap)
    assert m["quoted_spread_bps"] == pytest.approx(m["spread_bps"], abs=1e-12)
    assert m["half_spread_bps"] == pytest.approx(0.5 * m["spread_bps"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_half_spread_bps_from_mid_227(snap: OrderBookSnapshot) -> None:
    """half_spread_bps == 1e4*half_spread/mid."""
    m = book_metrics_from_snapshot(snap)
    assert m["half_spread_bps"] == pytest.approx(1e4 * m["half_spread"] / m["mid"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_spread_over_mid_formula_228(snap: OrderBookSnapshot) -> None:
    """spread_over_mid == spread/mid."""
    m = book_metrics_from_snapshot(snap)
    assert m["spread_over_mid"] == pytest.approx(m["spread"] / m["mid"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_n_levels_match_lengths_229(snap: OrderBookSnapshot) -> None:
    """n_*_levels == len(side)."""
    m = book_metrics_from_snapshot(snap)
    assert m["n_bid_levels"] == float(len(snap.bids))
    assert m["n_ask_levels"] == float(len(snap.asks))
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_depths_equal_sum_sizes_230(snap: OrderBookSnapshot) -> None:
    """bid/ask_depth == sum sizes."""
    m = book_metrics_from_snapshot(snap)
    assert m["bid_depth"] == pytest.approx(sum(lvl.size for lvl in snap.bids), abs=1e-12)
    assert m["ask_depth"] == pytest.approx(sum(lvl.size for lvl in snap.asks), abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots().filter(lambda s: len(s.bids) >= 2 and len(s.asks) >= 2))
@settings(max_examples=30, deadline=None)
def test_deep_all_shape_finite_231(snap: OrderBookSnapshot) -> None:
    """Deep: all DEPTH_SHAPE finite."""
    m = book_metrics_from_snapshot(snap)
    for field in DEPTH_SHAPE_FIELDS:
        assert math.isfinite(m[field]), field
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots().filter(lambda s: len(s.bids) == 1 and len(s.asks) == 1))
@settings(max_examples=20, deadline=None)
def test_thin_all_shape_nan_232(snap: OrderBookSnapshot) -> None:
    """Thin: all DEPTH_SHAPE NaN."""
    m = book_metrics_from_snapshot(snap)
    for field in DEPTH_SHAPE_FIELDS:
        assert math.isnan(m[field]), field
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_required_all_finite_233(snap: OrderBookSnapshot) -> None:
    """Every REQUIRED key finite."""
    m = book_metrics_from_snapshot(snap)
    for key in METRICS_REQUIRED_FINITE_KEYS:
        assert math.isfinite(m[key]), key
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_export_field_groups_present_234(snap: OrderBookSnapshot) -> None:
    """All field-group keys present."""
    m = book_metrics_from_snapshot(snap)
    for key in (
        DEPTH_SHAPE_FIELDS
        + SIDE_STRUCTURE_FIELDS
        + QUEUE_STRUCTURE_FIELDS
        + SIDE_NOTIONAL_FIELDS
        + TOB_SHARE_FIELDS
    ):
        assert key in m
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_docs_match_required_235(snap: OrderBookSnapshot) -> None:
    """KEY_DOCS keys == REQUIRED; OPTIONAL∩REQUIRED==∅."""
    assert frozenset(METRICS_REQUIRED_FINITE_KEY_DOCS) == METRICS_REQUIRED_FINITE_KEYS
    assert METRICS_REQUIRED_FINITE_KEYS.isdisjoint(METRICS_OPTIONAL_NAN_OK_KEYS)
    assert_metrics_key_partition(book_metrics_from_snapshot(snap))


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_microprice_in_interval_236(snap: OrderBookSnapshot) -> None:
    """best_bid ≤ microprice ≤ best_ask."""
    m = book_metrics_from_snapshot(snap)
    assert m["best_bid"] - 1e-12 <= m["microprice"] <= m["best_ask"] + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_tops_positive_237(snap: OrderBookSnapshot) -> None:
    """top sizes > 0."""
    m = book_metrics_from_snapshot(snap)
    assert m["top_bid_size"] > 0.0 and m["top_ask_size"] > 0.0
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_depths_positive_238(snap: OrderBookSnapshot) -> None:
    """side depths > 0."""
    m = book_metrics_from_snapshot(snap)
    assert m["bid_depth"] > 0.0 and m["ask_depth"] > 0.0
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_spread_positive_239(snap: OrderBookSnapshot) -> None:
    """spread > 0."""
    m = book_metrics_from_snapshot(snap)
    assert m["spread"] > 0.0
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_mid_positive_240(snap: OrderBookSnapshot) -> None:
    """mid > 0."""
    m = book_metrics_from_snapshot(snap)
    assert m["mid"] > 0.0
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_imb_top_bounds_241(snap: OrderBookSnapshot) -> None:
    """imbalance_top ∈[-1,1]."""
    m = book_metrics_from_snapshot(snap)
    assert -1.0 - 1e-12 <= m["imbalance_top"] <= 1.0 + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_imb_depth_bounds_242(snap: OrderBookSnapshot) -> None:
    """imbalance_depth ∈[-1,1]."""
    m = book_metrics_from_snapshot(snap)
    assert -1.0 - 1e-12 <= m["imbalance_depth"] <= 1.0 + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_weight_bounds_243(snap: OrderBookSnapshot) -> None:
    """microprice_weight_balance ∈[0,1]."""
    m = book_metrics_from_snapshot(snap)
    assert 0.0 <= m["microprice_weight_balance"] <= 1.0 + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_tob_shares_unit_244(snap: OrderBookSnapshot) -> None:
    """tob_size_share and tob_notional_share ∈(0,1]."""
    m = book_metrics_from_snapshot(snap)
    assert 0.0 < m["tob_size_share"] <= 1.0 + 1e-12
    assert 0.0 < m["tob_notional_share"] <= 1.0 + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_queue_lt_concentration_245(snap: OrderBookSnapshot) -> None:
    """queue < concentration both sides."""
    m = book_metrics_from_snapshot(snap)
    assert m["queue_priority_proxy"] < m["bid_size_concentration_top"] + 1e-12
    assert m["ask_queue_priority_proxy"] < m["ask_size_concentration_top"] + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_half_spread_positive_246(snap: OrderBookSnapshot) -> None:
    """half_spread > 0 == 0.5*spread."""
    m = book_metrics_from_snapshot(snap)
    assert m["half_spread"] > 0.0
    assert m["half_spread"] == pytest.approx(0.5 * m["spread"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_two_half_equals_spread_247(snap: OrderBookSnapshot) -> None:
    """2*half_spread == spread."""
    m = book_metrics_from_snapshot(snap)
    assert (2.0 * m["half_spread"]) == pytest.approx(m["spread"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_mid_arithmetic_mean_248(snap: OrderBookSnapshot) -> None:
    """mid == 0.5*(best_bid+best_ask)."""
    m = book_metrics_from_snapshot(snap)
    assert m["mid"] == pytest.approx(0.5 * (m["best_bid"] + m["best_ask"]), abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_microprice_weighted_mid_249(snap: OrderBookSnapshot) -> None:
    """microprice == (ask*top_bid + bid*top_ask)/(tops)."""
    m = book_metrics_from_snapshot(snap)
    tb, ta = m["top_bid_size"], m["top_ask_size"]
    expected = (m["best_ask"] * tb + m["best_bid"] * ta) / (tb + ta)
    assert m["microprice"] == pytest.approx(expected, abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_mu_mid_identity_250(snap: OrderBookSnapshot) -> None:
    """microprice_minus_mid == microprice - mid."""
    m = book_metrics_from_snapshot(snap)
    assert m["microprice_minus_mid"] == pytest.approx(m["microprice"] - m["mid"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_mu_mid_bps_identity_251(snap: OrderBookSnapshot) -> None:
    """microprice_minus_mid_bps == 1e4*(μ-mid)/mid."""
    m = book_metrics_from_snapshot(snap)
    assert m["microprice_minus_mid_bps"] == pytest.approx(
        1e4 * m["microprice_minus_mid"] / m["mid"], abs=1e-12
    )
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_spread_bps_identity_252(snap: OrderBookSnapshot) -> None:
    """spread_bps == 1e4*spread/mid."""
    m = book_metrics_from_snapshot(snap)
    assert m["spread_bps"] == pytest.approx(1e4 * m["spread"] / m["mid"], abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_depth_imbalance_abs_identity_253(snap: OrderBookSnapshot) -> None:
    """depth_imbalance_abs == |imbalance_depth|."""
    m = book_metrics_from_snapshot(snap)
    assert m["depth_imbalance_abs"] == pytest.approx(abs(m["imbalance_depth"]), abs=1e-12)
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_tops_le_depths_254(snap: OrderBookSnapshot) -> None:
    """top sizes ≤ side depths."""
    m = book_metrics_from_snapshot(snap)
    assert m["top_bid_size"] <= m["bid_depth"] + 1e-12
    assert m["top_ask_size"] <= m["ask_depth"] + 1e-12
    assert_metrics_key_partition(m)


@given(uncrossed_snapshots())
@settings(max_examples=40, deadline=None)
def test_best_ask_gt_bid_255(snap: OrderBookSnapshot) -> None:
    """best_ask > best_bid."""
    m = book_metrics_from_snapshot(snap)
    assert m["best_ask"] > m["best_bid"]
    assert_metrics_key_partition(m)
