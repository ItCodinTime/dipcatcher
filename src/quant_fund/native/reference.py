"""NumPy / library reference for the optional native kernels.

These functions are the default path when ``quant_core`` is not built. They
call the existing research implementations so receipt math does not grow a
second Python formula. Batch helpers (`turnover_series`, `book_features`,
`hash_many`) are the NumPy equivalents of those loops.

Float contract
--------------
Bit-for-bit with the library (NaNs included): ``rolling_mean``, ``rolling_std``,
``simple_returns``, ``wealth_index``, ``hash_bytes``, ``hash_many``.

Within ``RTOL`` / ``ATOL`` (NumPy pairwise reductions vs sequential ``f64``):
``ema``, ``rsi``, ``bollinger``, ``turnover``, ``turnover_series``,
``book_features``.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

RTOL = 1e-12
ATOL = 1e-12

BOOK_FIELDS: tuple[str, ...] = (
    "mid",
    "spread",
    "microprice",
    "imbalance_top",
    "imbalance_depth",
    "bid_depth",
    "ask_depth",
    "bid_log_size_slope",
    "ask_log_size_slope",
    "bid_log_price_slope",
    "ask_log_price_slope",
    "bid_mean_log_tick_spacing",
    "ask_mean_log_tick_spacing",
)

FloatArray = NDArray[np.float64]


def _panel(values: ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim not in (1, 2):
        raise ValueError(f"{name} expects a 1-d series or a 2-d panel with one series per row")
    return arr


def rolling_mean(values: ArrayLike, window: int) -> FloatArray:
    """Causal rolling mean. NaN until the window fills; non-finite inputs propagate."""
    from quant_fund.lightspeed.ema import rolling_mean as impl

    arr = _panel(values, "rolling_mean")
    if arr.ndim == 1:
        return np.asarray(impl(arr, window), dtype=np.float64)
    if arr.shape[0] == 0:
        return np.full(arr.shape, np.nan, dtype=np.float64)
    rows = [np.asarray(impl(arr[i], window), dtype=np.float64) for i in range(arr.shape[0])]
    return np.vstack(rows)


def rolling_std(values: ArrayLike, window: int) -> FloatArray:
    """Population rolling std (ddof 0), NaN-padded. Window < 2 is all NaN."""
    from quant_fund.lightspeed.ema import rolling_std as impl

    arr = _panel(values, "rolling_std")
    if arr.ndim == 1:
        return np.asarray(impl(arr, window), dtype=np.float64)
    if arr.shape[0] == 0:
        return np.full(arr.shape, np.nan, dtype=np.float64)
    rows = [np.asarray(impl(arr[i], window), dtype=np.float64) for i in range(arr.shape[0])]
    return np.vstack(rows)


def _close_array(values: ArrayLike) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


def prepare_close(values: ArrayLike, window: int) -> tuple[FloatArray, int]:
    """Same checks as ``quant_fund.features.indicators`` before a native call."""
    from quant_fund.features.indicators import _as_vec, _check_window

    series = _as_vec(_close_array(values), "close", window)
    width = _check_window(window, series.size)
    return np.ascontiguousarray(series, dtype=np.float64), width


def ema(values: ArrayLike, window: int) -> FloatArray:
    from quant_fund.features.indicators import ema as impl

    return np.asarray(impl(_close_array(values), window), dtype=np.float64)


def rsi(values: ArrayLike, window: int) -> FloatArray:
    from quant_fund.features.indicators import rsi as impl

    return np.asarray(impl(_close_array(values), window), dtype=np.float64)


def bollinger(values: ArrayLike, window: int = 20, num_sd: float = 2.0) -> dict[str, FloatArray]:
    from quant_fund.features.indicators import bollinger as impl

    raw = impl(_close_array(values), window, num_sd)
    return {key: np.asarray(val, dtype=np.float64) for key, val in raw.items()}


def simple_returns(prices: ArrayLike) -> FloatArray:
    from quant_fund.hedge_lab.directional import simple_returns as impl

    arr = np.asarray(prices, dtype=np.float64)
    if arr.ndim not in (1, 2):
        raise ValueError("simple_returns expects a 1-d path or a 2-d (time, asset) panel")
    return np.asarray(impl(arr), dtype=np.float64)


def wealth_index(returns: ArrayLike) -> FloatArray:
    from quant_fund.metrics.returns import wealth_index as impl

    return np.asarray(impl(np.asarray(returns, dtype=np.float64)), dtype=np.float64)


def turnover(weights: ArrayLike, prev_weights: ArrayLike) -> float:
    from quant_fund.metrics.returns import turnover as impl

    return float(
        impl(np.asarray(weights, dtype=np.float64), np.asarray(prev_weights, dtype=np.float64))
    )


def turnover_series(weights: ArrayLike) -> FloatArray:
    """L1 turnover between consecutive rows. Index 0 is 0. Shape ``(T, N)``."""
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError("turnover_series expects a (time, asset) array")
    out = np.empty(w.shape[0], dtype=np.float64)
    if w.shape[0] == 0:
        return out
    out[0] = 0.0
    for i in range(1, w.shape[0]):
        out[i] = turnover(w[i], w[i - 1])
    return out


def _ols_log_slope(values: FloatArray) -> float:
    n = int(values.size)
    if n < 2:
        return float("nan")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        return float("nan")
    y = np.log(values)
    x_centered = np.arange(n, dtype=float) - 0.5 * (n - 1)
    denominator = (n * (n * n - 1)) / 12.0
    if denominator <= 1e-18:
        return float("nan")
    return float(np.dot(x_centered, y) / denominator)


def _mean_log_spacing(prices: FloatArray) -> float:
    if prices.size < 2:
        return float("nan")
    if not np.isfinite(prices).all():
        return float("nan")
    gaps = np.abs(np.diff(prices))
    if not np.isfinite(gaps).all() or np.any(gaps <= 0.0):
        return float("nan")
    return float(np.mean(np.log(gaps)))


def _book_row(
    bid_px: FloatArray, bid_sz: FloatArray, ask_px: FloatArray, ask_sz: FloatArray
) -> dict[str, float]:
    nan_row = {name: float("nan") for name in BOOK_FIELDS}
    sides = (bid_px, bid_sz, ask_px, ask_sz)
    if any(col.size == 0 for col in sides):
        return nan_row
    if any((not np.isfinite(col).all()) or np.any(col <= 0.0) for col in sides):
        return nan_row
    if float(bid_px[0]) >= float(ask_px[0]):
        return nan_row
    bid_depth = float(sum(float(x) for x in bid_sz))
    ask_depth = float(sum(float(x) for x in ask_sz))
    top_bid = float(bid_sz[0])
    top_ask = float(ask_sz[0])
    mid = 0.5 * (float(bid_px[0]) + float(ask_px[0]))
    spread = float(ask_px[0]) - float(bid_px[0])
    denom = top_bid + top_ask
    if denom <= 0.0 or not math.isfinite(denom):
        microprice = mid
    else:
        microprice = (float(ask_px[0]) * top_bid + float(bid_px[0]) * top_ask) / denom
    imbalance_top = (top_bid - top_ask) / denom
    depth_sum = bid_depth + ask_depth
    imbalance_depth = (bid_depth - ask_depth) / depth_sum if depth_sum > 0.0 else 0.0
    return {
        "mid": mid,
        "spread": spread,
        "microprice": microprice,
        "imbalance_top": imbalance_top,
        "imbalance_depth": imbalance_depth,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "bid_log_size_slope": _ols_log_slope(bid_sz),
        "ask_log_size_slope": _ols_log_slope(ask_sz),
        "bid_log_price_slope": _ols_log_slope(bid_px),
        "ask_log_price_slope": _ols_log_slope(ask_px),
        "bid_mean_log_tick_spacing": _mean_log_spacing(bid_px),
        "ask_mean_log_tick_spacing": _mean_log_spacing(ask_px),
    }


def _as_book(values: ArrayLike) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return np.ascontiguousarray(arr.reshape(1, -1))
    if arr.ndim != 2:
        raise ValueError("book arrays must be 1-d or 2-d")
    return np.ascontiguousarray(arr)


def book_features(
    bid_px: ArrayLike,
    bid_sz: ArrayLike,
    ask_px: ArrayLike,
    ask_sz: ArrayLike,
) -> dict[str, FloatArray]:
    """Per-snapshot book features. Invalid rows are NaN, not exceptions."""
    bp = _as_book(bid_px)
    bs = _as_book(bid_sz)
    ap = _as_book(ask_px)
    az = _as_book(ask_sz)
    shape = bp.shape
    if bs.shape != shape or ap.shape != shape or az.shape != shape:
        raise ValueError("book arrays must share a shape")
    if shape[1] < 1:
        raise ValueError("book depth must be >= 1")
    rows = [_book_row(bp[i], bs[i], ap[i], az[i]) for i in range(shape[0])]
    return {
        name: np.asarray([row[name] for row in rows], dtype=np.float64)
        if rows
        else np.asarray([], dtype=np.float64)
        for name in BOOK_FIELDS
    }


def hash_bytes(data: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(bytes(data)).hexdigest()


def hash_many(chunks: Sequence[bytes | bytearray | memoryview]) -> list[str]:
    return [hash_bytes(chunk) for chunk in chunks]


def as_bytes(data: Any) -> bytes:
    if isinstance(data, bytes):
        return data
    return bytes(data)
