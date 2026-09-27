"""Optional native acceleration for the hottest numeric kernels.

``QUANT_FUND_NATIVE`` is read once, at import:

- ``auto`` (default): import ``quant_core`` if it is installed, otherwise NumPy.
- ``python``: always the NumPy / library reference.
- ``rust``: require ``quant_core`` (``ImportError`` when the extension is absent).

Changing the variable requires a new interpreter. The reference path is what
CI runs. Building the extension does not change committed research numbers
unless this process was started with the extension importable.

Kernels
-------
Rolling windows (``rolling_mean``, ``rolling_std``, ``ema``, ``rsi``,
``bollinger``), return / turnover accounting (``simple_returns``,
``wealth_index``, ``turnover``, ``turnover_series``), order-book features
(``book_features``), and SHA-256 (``hash_bytes``, ``hash_many``).

Tolerances live on ``quant_fund.native.reference`` (``RTOL``, ``ATOL``).
Cumsum rolling moments, wealth, simple returns, and SHA-256 match bit-for-bit.
EMA, RSI, Bollinger, turnover sums, and book OLS match within that tolerance
because NumPy's pairwise reductions are not the sequential ``f64`` sum.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quant_fund.native.reference import (
    ATOL,
    BOOK_FIELDS,
    RTOL,
    as_bytes,
    prepare_close,
)
from quant_fund.native.reference import (
    bollinger as bollinger_py,
)
from quant_fund.native.reference import (
    book_features as book_features_py,
)
from quant_fund.native.reference import (
    ema as ema_py,
)
from quant_fund.native.reference import (
    hash_bytes as hash_bytes_py,
)
from quant_fund.native.reference import (
    hash_many as hash_many_py,
)
from quant_fund.native.reference import (
    rolling_mean as rolling_mean_py,
)
from quant_fund.native.reference import (
    rolling_std as rolling_std_py,
)
from quant_fund.native.reference import (
    rsi as rsi_py,
)
from quant_fund.native.reference import (
    simple_returns as simple_returns_py,
)
from quant_fund.native.reference import (
    turnover as turnover_py,
)
from quant_fund.native.reference import (
    turnover_series as turnover_series_py,
)
from quant_fund.native.reference import (
    wealth_index as wealth_index_py,
)

FloatArray = NDArray[np.float64]

_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1


def _flag() -> str:
    raw = os.environ.get("QUANT_FUND_NATIVE", "auto")
    flag = raw.strip().lower()
    if flag in {"", "auto"}:
        return "auto"
    if flag in {"python", "py", "numpy"}:
        return "python"
    if flag in {"rust", "native"}:
        return "rust"
    raise ValueError(f"QUANT_FUND_NATIVE must be auto, python, or rust (got {raw!r})")


def _load_rust(flag: str) -> Any:
    if flag == "python":
        return None
    try:
        import quant_core
    except ImportError as exc:
        if flag == "rust":
            raise ImportError(
                "QUANT_FUND_NATIVE=rust but the quant_core extension is not "
                "importable. From the repo root: make native"
            ) from exc
        return None
    return quant_core


_FLAG = _flag()
_RUST: Any = _load_rust(_FLAG)
BACKEND: str = "rust" if _RUST is not None else "python"


def _width(window: int) -> int | None:
    width = int(window)
    if width < _I64_MIN or width > _I64_MAX:
        return None
    return width


def _f64(values: ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim not in (1, 2):
        raise ValueError(f"{name} expects a 1-d series or a 2-d panel with one series per row")
    return np.ascontiguousarray(arr)


def rolling_mean(values: ArrayLike, window: int) -> FloatArray:
    if _RUST is None:
        return rolling_mean_py(values, window)
    arr = _f64(values, "rolling_mean")
    width = _width(window)
    if width is None:
        return rolling_mean_py(arr, window)
    if arr.ndim == 1:
        return np.asarray(_RUST.rolling_mean(arr, width), dtype=np.float64)
    flat = np.ascontiguousarray(arr.reshape(-1))
    out = np.asarray(
        _RUST.rolling_mean_2d(flat, arr.shape[0], arr.shape[1], width),
        dtype=np.float64,
    )
    return out.reshape(arr.shape)


def rolling_std(values: ArrayLike, window: int) -> FloatArray:
    if _RUST is None:
        return rolling_std_py(values, window)
    arr = _f64(values, "rolling_std")
    width = _width(window)
    if width is None:
        return rolling_std_py(arr, window)
    if arr.ndim == 1:
        return np.asarray(_RUST.rolling_std(arr, width), dtype=np.float64)
    flat = np.ascontiguousarray(arr.reshape(-1))
    out = np.asarray(
        _RUST.rolling_std_2d(flat, arr.shape[0], arr.shape[1], width),
        dtype=np.float64,
    )
    return out.reshape(arr.shape)


def ema(values: ArrayLike, window: int) -> FloatArray:
    if _RUST is None:
        return ema_py(values, window)
    series, width = prepare_close(values, window)
    return np.asarray(_RUST.ema(series, width), dtype=np.float64)


def rsi(values: ArrayLike, window: int) -> FloatArray:
    if _RUST is None:
        return rsi_py(values, window)
    series, width = prepare_close(values, window)
    return np.asarray(_RUST.rsi(series, width), dtype=np.float64)


def bollinger(
    values: ArrayLike,
    window: int = 20,
    num_sd: float = 2.0,
) -> dict[str, FloatArray]:
    if _RUST is None:
        return bollinger_py(values, window, num_sd)
    series, width = prepare_close(values, window)
    if not np.isfinite(num_sd) or float(num_sd) <= 0.0:
        raise ValueError("num_sd must be positive")
    raw = _RUST.bollinger(series, width, float(num_sd))
    return {key: np.asarray(raw[key], dtype=np.float64) for key in raw}


def simple_returns(prices: ArrayLike) -> FloatArray:
    if _RUST is None:
        return simple_returns_py(prices)
    arr = np.asarray(prices, dtype=np.float64)
    if arr.ndim not in (1, 2):
        raise ValueError("simple_returns expects a 1-d path or a 2-d (time, asset) panel")
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 1:
        return np.asarray(_RUST.simple_returns(arr), dtype=np.float64)
    flat = np.ascontiguousarray(arr.reshape(-1))
    out = np.asarray(_RUST.simple_returns_2d(flat, arr.shape[0], arr.shape[1]), dtype=np.float64)
    return out.reshape(arr.shape)


def wealth_index(returns: ArrayLike) -> FloatArray:
    if _RUST is None:
        return wealth_index_py(returns)
    series = np.ascontiguousarray(np.asarray(returns, dtype=np.float64).reshape(-1))
    return np.asarray(_RUST.wealth_index(series), dtype=np.float64)


def turnover(weights: ArrayLike, prev_weights: ArrayLike) -> float:
    if _RUST is None:
        return turnover_py(weights, prev_weights)
    w = np.ascontiguousarray(np.asarray(weights, dtype=np.float64).reshape(-1))
    p = np.ascontiguousarray(np.asarray(prev_weights, dtype=np.float64).reshape(-1))
    if w.shape != p.shape:
        raise ValueError(f"turnover weight length mismatch: {w.shape[0]} vs {p.shape[0]}")
    return float(_RUST.turnover(w, p))


def turnover_series(weights: ArrayLike) -> FloatArray:
    if _RUST is None:
        return turnover_series_py(weights)
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError("turnover_series expects a (time, asset) array")
    w = np.ascontiguousarray(w)
    flat = np.ascontiguousarray(w.reshape(-1))
    return np.asarray(_RUST.turnover_series(flat, w.shape[0], w.shape[1]), dtype=np.float64)


def book_features(
    bid_px: ArrayLike,
    bid_sz: ArrayLike,
    ask_px: ArrayLike,
    ask_sz: ArrayLike,
) -> dict[str, FloatArray]:
    if _RUST is None:
        return book_features_py(bid_px, bid_sz, ask_px, ask_sz)
    panels = []
    for values in (bid_px, bid_sz, ask_px, ask_sz):
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError("book arrays must be 1-d or 2-d")
        panels.append(np.ascontiguousarray(arr))
    shape = panels[0].shape
    if any(panel.shape != shape for panel in panels[1:]):
        raise ValueError("book arrays must share a shape")
    if shape[1] < 1:
        raise ValueError("book depth must be >= 1")
    flats = [np.ascontiguousarray(panel.reshape(-1)) for panel in panels]
    raw = _RUST.book_features(*flats, shape[0], shape[1])
    return {key: np.asarray(raw[key], dtype=np.float64) for key in BOOK_FIELDS}


def hash_bytes(data: bytes | bytearray | memoryview) -> str:
    blob = as_bytes(data)
    if _RUST is None:
        return hash_bytes_py(blob)
    return str(_RUST.hash_bytes(blob))


def hash_many(chunks: Any) -> list[str]:
    payload = [as_bytes(chunk) for chunk in chunks]
    if _RUST is None:
        return hash_many_py(payload)
    return [str(item) for item in _RUST.hash_many(payload)]


__all__ = [
    "ATOL",
    "BACKEND",
    "BOOK_FIELDS",
    "RTOL",
    "bollinger",
    "book_features",
    "ema",
    "hash_bytes",
    "hash_many",
    "rolling_mean",
    "rolling_std",
    "rsi",
    "simple_returns",
    "turnover",
    "turnover_series",
    "wealth_index",
]
