"""Parity of quant_fund.native against the NumPy reference.

SYNTHETIC inputs only. When ``quant_core`` is installed the public functions
take the Rust path; otherwise they are the reference. Both must agree with
the documented contract (bit-for-bit, or ``RTOL``/``ATOL``).
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund import native
from quant_fund.features.indicators import bollinger as bollinger_lib
from quant_fund.features.indicators import ema as ema_lib
from quant_fund.features.indicators import rsi as rsi_lib
from quant_fund.hedge_lab.directional import simple_returns as simple_returns_lib
from quant_fund.lightspeed.ema import rolling_mean as rolling_mean_lib
from quant_fund.lightspeed.ema import rolling_std as rolling_std_lib
from quant_fund.metrics.returns import turnover as turnover_lib
from quant_fund.metrics.returns import wealth_index as wealth_index_lib
from quant_fund.microstructure.book_metrics import book_metrics_from_snapshot
from quant_fund.native.reference import (
    ATOL,
    BOOK_FIELDS,
    RTOL,
)
from quant_fund.native.reference import (
    book_features as book_features_ref,
)
from quant_fund.schemas.order_book import BookLevel, OrderBookSnapshot

_FINITE = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False, width=64)
_WEIRD = st.one_of(
    _FINITE,
    st.just(float("nan")),
    st.just(float("inf")),
    st.just(float("-inf")),
    st.just(0.0),
    st.just(-0.0),
)


def _bit_exact(actual: np.ndarray, expected: np.ndarray) -> None:
    got = np.asarray(actual, dtype=np.float64)
    exp = np.asarray(expected, dtype=np.float64)
    assert got.shape == exp.shape
    nan_mask = np.isnan(got) & np.isnan(exp)
    assert np.array_equal(np.where(nan_mask, 0.0, got), np.where(nan_mask, 0.0, exp))


def _close(actual: np.ndarray, expected: np.ndarray) -> None:
    got = np.asarray(actual, dtype=np.float64)
    exp = np.asarray(expected, dtype=np.float64)
    assert got.shape == exp.shape
    assert np.allclose(got, exp, rtol=RTOL, atol=ATOL, equal_nan=True)


def _raise_message(fn: object) -> str:
    with pytest.raises(ValueError) as caught:
        fn()  # type: ignore[operator]
    return str(caught.value)


@given(st.lists(_WEIRD, max_size=24), st.integers(-2, 30))
@settings(max_examples=40, deadline=None)
def test_rolling_mean_bit_exact(values: list[float], window: int) -> None:
    series = np.asarray(values, dtype=np.float64)
    expected = rolling_mean_lib(series, window)
    _bit_exact(native.rolling_mean(series, window), expected)
    if series.size:
        panel = np.vstack([series, series[::-1]])
        got = native.rolling_mean(panel, window)
        _bit_exact(got[0], expected)
        _bit_exact(got[1], rolling_mean_lib(series[::-1], window))


@given(st.lists(_WEIRD, max_size=24), st.integers(-2, 30))
@settings(max_examples=40, deadline=None)
def test_rolling_std_bit_exact(values: list[float], window: int) -> None:
    series = np.asarray(values, dtype=np.float64)
    expected = rolling_std_lib(series, window)
    _bit_exact(native.rolling_std(series, window), expected)


@given(st.lists(_FINITE, min_size=1, max_size=32), st.integers(1, 16))
@settings(max_examples=30, deadline=None)
def test_ema_rsi_bollinger_close(values: list[float], window: int) -> None:
    series = np.asarray(values, dtype=np.float64)
    if window > series.size:
        assert _raise_message(lambda: native.ema(series, window)) == _raise_message(
            lambda: ema_lib(series, window)
        )
        return
    _close(native.ema(series, window), ema_lib(series, window))
    _close(native.rsi(series, window), rsi_lib(series, window))
    got = native.bollinger(series, window, 2.0)
    exp = bollinger_lib(series, window, 2.0)
    assert set(got) == set(exp)
    for key in exp:
        _close(got[key], exp[key])


@given(st.lists(_WEIRD, max_size=20))
@settings(max_examples=30, deadline=None)
def test_simple_returns_and_wealth_bit_exact(values: list[float]) -> None:
    prices = np.asarray(values, dtype=np.float64)
    _bit_exact(native.simple_returns(prices), simple_returns_lib(prices))
    if prices.size >= 2:
        panel = np.column_stack([prices, prices + 1.0])
        _bit_exact(native.simple_returns(panel), simple_returns_lib(panel))
    returns = np.asarray(values, dtype=np.float64)
    _bit_exact(native.wealth_index(returns), wealth_index_lib(returns))


@given(
    st.lists(_FINITE, max_size=12),
    st.lists(_FINITE, max_size=12),
)
@settings(max_examples=30, deadline=None)
def test_turnover_close(left: list[float], right: list[float]) -> None:
    w = np.asarray(left, dtype=np.float64)
    p = np.asarray(right, dtype=np.float64)
    if w.shape != p.shape:
        assert _raise_message(lambda: native.turnover(w, p)) == _raise_message(
            lambda: turnover_lib(w, p)
        )
        return
    assert native.turnover(w, p) == pytest.approx(
        turnover_lib(w, p), rel=RTOL, abs=ATOL, nan_ok=True
    )


@given(
    st.integers(0, 6),
    st.integers(0, 5),
    st.lists(_FINITE, max_size=30),
)
@settings(max_examples=25, deadline=None)
def test_turnover_series_close(rows: int, cols: int, values: list[float]) -> None:
    need = rows * cols
    data = list(values) + [0.0] * need
    weights = np.asarray(data[:need], dtype=np.float64).reshape(rows, cols)
    got = native.turnover_series(weights)
    if rows == 0:
        assert got.shape == (0,)
        return
    assert got[0] == 0.0
    for i in range(1, rows):
        assert got[i] == pytest.approx(turnover_lib(weights[i], weights[i - 1]), rel=RTOL, abs=ATOL)


def test_edges_empty_nan_inf_and_indicator_errors() -> None:
    empty = np.asarray([], dtype=np.float64)
    _bit_exact(native.rolling_mean(empty, 3), rolling_mean_lib(empty, 3))
    _bit_exact(native.rolling_std(empty, 3), rolling_std_lib(empty, 3))
    _bit_exact(native.rolling_mean(empty, 0), rolling_mean_lib(empty, 0))
    _bit_exact(
        native.rolling_std(np.asarray([1.0, 2.0]), 1), rolling_std_lib(np.asarray([1.0, 2.0]), 1)
    )
    _bit_exact(native.wealth_index(empty), wealth_index_lib(empty))
    assert native.turnover(empty, empty) == 0.0
    poisoned = np.asarray([1.0, np.nan, 3.0, 4.0, np.inf, 6.0], dtype=np.float64)
    _bit_exact(native.rolling_mean(poisoned, 2), rolling_mean_lib(poisoned, 2))
    _bit_exact(native.rolling_std(poisoned, 2), rolling_std_lib(poisoned, 2))
    assert np.isnan(native.turnover(np.asarray([1.0, np.nan]), np.asarray([0.0, 0.0])))
    with pytest.raises(ValueError, match="positive"):
        native.bollinger(np.asarray([1.0, 2.0, 3.0, 4.0]), 2, 0.0)
    assert _raise_message(lambda: native.rsi(np.asarray([1.0, np.nan, 3.0]), 2)) == _raise_message(
        lambda: rsi_lib(np.asarray([1.0, np.nan, 3.0]), 2)
    )
    nice = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    _bit_exact(native.ema(nice, 3), np.asarray([np.nan, np.nan, 2.0, 3.0, 4.0]))
    panel = np.asarray([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    _bit_exact(native.rolling_mean(np.asfortranarray(panel), 2), rolling_mean_lib_rows(panel, 2))


def rolling_mean_lib_rows(panel: np.ndarray, window: int) -> np.ndarray:
    return np.vstack([rolling_mean_lib(row, window) for row in panel])


def test_hash_matches_hashlib_including_empty() -> None:
    import hashlib

    empty = hashlib.sha256(b"").hexdigest()
    assert empty == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert native.hash_bytes(b"") == empty
    blobs = [b"", b"abc", b"\x00\xff", "café".encode()]
    assert native.hash_many(blobs) == [hashlib.sha256(blob).hexdigest() for blob in blobs]
    assert native.hash_many([]) == []


def _snapshot(
    bid_px: list[float], bid_sz: list[float], ask_px: list[float], ask_sz: list[float]
) -> OrderBookSnapshot:
    now = datetime(2024, 1, 2, tzinfo=UTC)
    return OrderBookSnapshot(
        security_id="S",
        event_time=now,
        available_time=now,
        bids=[BookLevel(price=p, size=s) for p, s in zip(bid_px, bid_sz, strict=True)],
        asks=[BookLevel(price=p, size=s) for p, s in zip(ask_px, ask_sz, strict=True)],
        depth=len(bid_px),
    )


@given(
    st.integers(1, 6),
    st.lists(_FINITE, min_size=8, max_size=8),
)
@settings(max_examples=25, deadline=None)
def test_book_features_match_snapshot_metrics(depth: int, draws: list[float]) -> None:
    # Fixed tick grid so the book stays strictly sorted. Sizes stay positive.
    bid_px = np.asarray([100.0 - 0.5 * i for i in range(depth)], dtype=np.float64)
    ask_px = np.asarray([101.0 + 0.5 * i for i in range(depth)], dtype=np.float64)
    bid_sz = np.asarray([1.0 + abs(draws[i]) for i in range(depth)], dtype=np.float64)
    ask_sz = np.asarray([1.0 + abs(draws[i + 1]) for i in range(depth)], dtype=np.float64)
    snap = _snapshot(bid_px.tolist(), bid_sz.tolist(), ask_px.tolist(), ask_sz.tolist())
    metrics = book_metrics_from_snapshot(snap)
    got = native.book_features(bid_px, bid_sz, ask_px, ask_sz)
    ref = book_features_ref(bid_px, bid_sz, ask_px, ask_sz)
    for name in BOOK_FIELDS:
        _close(got[name], ref[name])
        assert float(ref[name][0]) == pytest.approx(
            float(metrics[name]), rel=0.0, abs=0.0, nan_ok=True
        )


def test_book_invalid_rows_are_nan_and_depth_one_slopes_are_nan() -> None:
    bid_px = np.asarray([[10.0, 9.0], [10.0, 9.0], [10.0, 9.0]], dtype=np.float64)
    ask_px = np.asarray([[11.0, 12.0], [10.0, 12.0], [11.0, 12.0]], dtype=np.float64)
    bid_sz = np.asarray([[1.0, 1.0], [1.0, 1.0], [1.0, np.nan]], dtype=np.float64)
    ask_sz = np.asarray([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    got = native.book_features(bid_px, bid_sz, ask_px, ask_sz)
    assert np.isfinite(got["mid"][0])
    assert np.isnan(got["mid"][1])
    assert np.isnan(got["spread"][2])
    thin = native.book_features(
        np.asarray([10.0]), np.asarray([2.0]), np.asarray([11.0]), np.asarray([3.0])
    )
    assert thin["mid"][0] == pytest.approx(10.5)
    assert np.isnan(thin["bid_log_size_slope"][0])
    assert np.isnan(thin["ask_mean_log_tick_spacing"][0])
    empty = native.book_features(
        np.zeros((0, 2)),
        np.zeros((0, 2)),
        np.zeros((0, 2)),
        np.zeros((0, 2)),
    )
    assert empty["mid"].shape == (0,)
    with pytest.raises(ValueError, match="shape"):
        native.book_features(np.ones((2, 2)), np.ones((2, 3)), np.ones((2, 2)), np.ones((2, 2)))
