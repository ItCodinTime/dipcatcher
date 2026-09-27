"""Information-driven bar sampling (AFML ch. 2; Easley-O'Hara microstructure).

Alternative bar constructors that sample on activity rather than clock time -
they produce more iid/Gaussian return distributions and allocate sampling to
periods of information arrival.  Inputs are tick arrays (price, size) ordered
by time; outputs are bar boundary indices plus per-bar aggregates.

References:
- Lopez de Prado (2018). *Advances in Financial Machine Learning*, ch. 2 -
    tick/volume/dollar bars, imbalance bars (TIB/VIB/DIB), run bars (TRB/VRB/DRB).
- Easley, Lopez de Prado, O'Hara (2012). Flow toxicity and liquidity in a
  high-frequency world (VPIN/imbalance motivation). *RFS* 25.
- Easley, Kiefer, O'Hara, Paperman (1996). Liquidity, information, and
  infrequently traded stocks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.features._kernels import imbalance_bounds, reset_bounds, run_bounds

Array = NDArray[np.float64]
IdxArray = NDArray[np.intp]


def _as_ticks(prices: Array, sizes: Array) -> tuple[Array, Array]:
    p = np.asarray(prices, dtype=float).reshape(-1)
    s = np.asarray(sizes, dtype=float).reshape(-1)
    if p.size != s.size or p.size < 2:
        raise ValueError("prices and sizes must be non-empty equal-length")
    if not np.isfinite(p).all() or not np.isfinite(s).all():
        raise ValueError("prices and sizes must be finite")
    if np.any(p <= 0.0) or np.any(s < 0.0):
        raise ValueError("prices must be positive; sizes non-negative")
    return p, s


def _ohlcv(p: Array, s: Array, bounds: IdxArray) -> dict[str, Array]:
    """Aggregate ticks into OHLCV per boundary.

    ``bounds`` are bar-start indices. Each bar runs to the next start, and the
    last bar runs through the series end. High/low/sums use ``reduceat`` /
    slice indexing, which matches the per-bar slice reductions when the starts
    are strictly increasing.
    """
    idx = np.asarray(bounds, dtype=np.intp)
    ends = np.empty(idx.shape[0], dtype=np.intp)
    ends[:-1] = idx[1:]
    ends[-1] = p.size
    dollar = p * s
    return {
        "start": idx.astype(float),
        "open": p[idx],
        "high": np.maximum.reduceat(p, idx),
        "low": np.minimum.reduceat(p, idx),
        "close": p[ends - 1],
        "volume": np.add.reduceat(s, idx),
        "dollar": np.add.reduceat(dollar, idx),
    }


def _bounds_or_raise(bounds: np.ndarray, message: str) -> IdxArray:
    if bounds.size == 0:
        raise ValueError(message)
    return np.asarray(bounds, dtype=np.intp)


def tick_bars(prices: Array, sizes: Array, ticks_per_bar: int = 10) -> dict[str, Array]:
    """Fixed-count tick bars (AFML 2.3.1)."""
    p, s = _as_ticks(prices, sizes)
    if isinstance(ticks_per_bar, bool) or not isinstance(ticks_per_bar, int) or ticks_per_bar < 1:
        raise ValueError("ticks_per_bar must be a positive integer")
    n_bars = p.size // ticks_per_bar
    if n_bars < 1:
        raise ValueError("not enough ticks for one bar")
    bounds = np.arange(0, n_bars * ticks_per_bar, ticks_per_bar, dtype=np.intp)
    return _ohlcv(p, s, bounds)


def volume_bars(prices: Array, sizes: Array, volume_per_bar: float) -> dict[str, Array]:
    """Volume bars: close a bar when cumulative volume crosses the threshold."""
    p, s = _as_ticks(prices, sizes)
    if not np.isfinite(volume_per_bar) or volume_per_bar <= 0.0:
        raise ValueError("volume_per_bar must be positive and finite")
    bounds = _bounds_or_raise(
        reset_bounds(np.ascontiguousarray(s, dtype=np.float64), float(volume_per_bar)),
        "no complete volume bars",
    )
    return _ohlcv(p, s, bounds)


def dollar_bars(prices: Array, sizes: Array, dollar_per_bar: float) -> dict[str, Array]:
    """Dollar bars: close a bar when ``sum(price * size)`` crosses the threshold."""
    p, s = _as_ticks(prices, sizes)
    if not np.isfinite(dollar_per_bar) or dollar_per_bar <= 0.0:
        raise ValueError("dollar_per_bar must be positive and finite")
    weights = np.ascontiguousarray(p * s, dtype=np.float64)
    bounds = _bounds_or_raise(
        reset_bounds(weights, float(dollar_per_bar)),
        "no complete dollar bars",
    )
    return _ohlcv(p, s, bounds)


def _tick_signs(p: Array) -> Array:
    """Lee-Ready-style tick-rule signs: +/-1 on price moves, carry on zero.

    The forward fill copies a leading zero through the flat prefix. The
    ``signs[0]`` assignment happens after that fill, so only index 0 changes
    and the copied zeros stay zero.
    """
    d = np.diff(p, prepend=p[0])
    signs = np.sign(d)
    nonzero = signs != 0.0
    idx = np.where(nonzero, np.arange(signs.size), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = signs[idx]
    filled = np.array(filled, dtype=float, copy=True)
    filled[0] = 1.0 if signs[0] >= 0.0 else -1.0
    return filled


def _expect_ewma(expected_ticks: int, alpha_ewma: float) -> float:
    if expected_ticks < 2 or not np.isfinite(alpha_ewma) or not (0.0 < alpha_ewma <= 1.0):
        raise ValueError("expected_ticks >= 2 and alpha_ewma in (0,1] required")
    return float(expected_ticks)


def tick_imbalance_bars(
    prices: Array, sizes: Array, expected_ticks: int = 50, alpha_ewma: float = 0.1
) -> dict[str, Array]:
    """Tick imbalance bars (AFML 2.5.2): close when |sum of tick signs| exceeds
    ``E[ticks] * E[|imbalance|]`` with EWMA-updated expectations."""
    p, s = _as_ticks(prices, sizes)
    e_ticks = _expect_ewma(expected_ticks, alpha_ewma)
    b = np.ascontiguousarray(_tick_signs(p), dtype=np.float64)
    bounds = _bounds_or_raise(
        imbalance_bounds(b, float(alpha_ewma), e_ticks, 1.0),
        "no complete imbalance bars",
    )
    return _ohlcv(p, s, bounds)


def tick_run_bars(
    prices: Array, sizes: Array, expected_ticks: int = 50, alpha_ewma: float = 0.1
) -> dict[str, Array]:
    """Tick run bars (AFML 2.5.2): close when the max run of same-sign ticks
    exceeds ``E[T] * E[max-buy-fraction]`` - captures one-sided flow."""
    p, s = _as_ticks(prices, sizes)
    e_ticks = _expect_ewma(expected_ticks, alpha_ewma)
    b = np.ascontiguousarray(_tick_signs(p), dtype=np.float64)
    bounds = _bounds_or_raise(
        run_bounds(b, float(alpha_ewma), e_ticks),
        "no complete run bars",
    )
    return _ohlcv(p, s, bounds)


def dollar_imbalance_bars(
    prices: Array, sizes: Array, expected_ticks: int = 50, alpha_ewma: float = 0.1
) -> dict[str, Array]:
    """Dollar imbalance bars: close when ``|sum(b_t * dollar_t)|`` exceeds the
    EWMA-scaled threshold (AFML 2.5.2, dollar variant)."""
    p, s = _as_ticks(prices, sizes)
    e_ticks = _expect_ewma(expected_ticks, alpha_ewma)
    b = _tick_signs(p)
    dollar = np.ascontiguousarray(p * s * b, dtype=np.float64)
    e_init = float(np.abs(dollar).mean())
    bounds = _bounds_or_raise(
        imbalance_bounds(dollar, float(alpha_ewma), e_ticks, e_init),
        "no complete dollar-imbalance bars",
    )
    return _ohlcv(p, s, bounds)
