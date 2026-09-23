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
    """Aggregate ticks into OHLCV per boundary (bounds = bar start indices,
    implicitly closed at the next bound / series end)."""
    starts = np.append(bounds, p.size)
    o = np.array([p[a] for a in bounds])
    c = np.array([p[b - 1] for a, b in zip(bounds, starts[1:], strict=True)])
    hi = np.array([p[a:b].max() for a, b in zip(bounds, starts[1:], strict=True)])
    lo = np.array([p[a:b].min() for a, b in zip(bounds, starts[1:], strict=True)])
    v = np.array([s[a:b].sum() for a, b in zip(bounds, starts[1:], strict=True)])
    d = np.array([(p[a:b] * s[a:b]).sum() for a, b in zip(bounds, starts[1:], strict=True)])
    return {
        "start": bounds.astype(float),
        "open": o,
        "high": hi,
        "low": lo,
        "close": c,
        "volume": v,
        "dollar": d,
    }


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
    starts: list[int] = []
    acc = 0.0
    start = 0
    for t in range(p.size):
        acc += s[t]
        if acc >= volume_per_bar:
            starts.append(start)
            start = t + 1
            acc = 0.0
    bounds = np.asarray([st for st in starts if st < p.size - 1], dtype=np.intp)
    if bounds.size == 0:
        raise ValueError("no complete volume bars")
    return _ohlcv(p, s, bounds)


def dollar_bars(prices: Array, sizes: Array, dollar_per_bar: float) -> dict[str, Array]:
    """Dollar bars: close a bar when ``sum(price * size)`` crosses the threshold."""
    p, s = _as_ticks(prices, sizes)
    if not np.isfinite(dollar_per_bar) or dollar_per_bar <= 0.0:
        raise ValueError("dollar_per_bar must be positive and finite")
    starts: list[int] = []
    acc = 0.0
    start = 0
    for t in range(p.size):
        acc += p[t] * s[t]
        if acc >= dollar_per_bar:
            starts.append(start)
            start = t + 1
            acc = 0.0
    bounds = np.asarray([st for st in starts if st < p.size - 1], dtype=np.intp)
    if bounds.size == 0:
        raise ValueError("no complete dollar bars")
    return _ohlcv(p, s, bounds)


def _tick_signs(p: Array) -> Array:
    """Lee-Ready-style tick-rule signs: +/-1 on price moves, carry on zero."""
    d = np.diff(p, prepend=p[0])
    signs = np.sign(d)
    # Carry last nonzero sign forward on unchanged prices.
    for t in range(1, signs.size):
        if signs[t] == 0.0:
            signs[t] = signs[t - 1]
    signs[0] = 1.0 if signs[0] >= 0 else -1.0
    return signs


def tick_imbalance_bars(
    prices: Array, sizes: Array, expected_ticks: int = 50, alpha_ewma: float = 0.1
) -> dict[str, Array]:
    """Tick imbalance bars (AFML 2.5.2): close when |sum of tick signs| exceeds
    ``E[ticks] * E[|imbalance|]`` with EWMA-updated expectations."""
    p, s = _as_ticks(prices, sizes)
    if expected_ticks < 2 or not np.isfinite(alpha_ewma) or not (0.0 < alpha_ewma <= 1.0):
        raise ValueError("expected_ticks >= 2 and alpha_ewma in (0,1] required")
    b = _tick_signs(p)
    starts: list[int] = []
    theta = 0.0
    e_theta = 1.0  # EWMA of |imbalance| per tick
    start = 0
    e_ticks = float(expected_ticks)
    for t in range(p.size):
        theta += b[t]
        e_theta = (1.0 - alpha_ewma) * e_theta + alpha_ewma * abs(b[t])
        # Trigger: |theta_T| >= E[T] * E[|b_t|] (AFML 2.5.2).
        if abs(theta) >= e_ticks * max(e_theta, 1e-12):
            starts.append(start)
            start = t + 1
            theta = 0.0
    bounds = np.asarray([st for st in starts if st < p.size - 1], dtype=np.intp)
    if bounds.size == 0:
        raise ValueError("no complete imbalance bars")
    return _ohlcv(p, s, bounds)


def tick_run_bars(
    prices: Array, sizes: Array, expected_ticks: int = 50, alpha_ewma: float = 0.1
) -> dict[str, Array]:
    """Tick run bars (AFML 2.5.2): close when the max run of same-sign ticks
    exceeds ``E[T] * E[max-buy-fraction]`` - captures one-sided flow."""
    p, s = _as_ticks(prices, sizes)
    if expected_ticks < 2 or not np.isfinite(alpha_ewma) or not (0.0 < alpha_ewma <= 1.0):
        raise ValueError("expected_ticks >= 2 and alpha_ewma in (0,1] required")
    b = _tick_signs(p)
    starts: list[int] = []
    start = 0
    e_share = 0.5  # EWMA of buy-share per tick
    e_ticks = float(expected_ticks)
    for t in range(p.size):
        is_buy = 1.0 if b[t] > 0 else 0.0
        e_share = (1.0 - alpha_ewma) * e_share + alpha_ewma * is_buy
        run_buy = float(np.sum(b[start : t + 1] > 0))
        run_sell = float(np.sum(b[start : t + 1] < 0))
        run = max(run_buy, run_sell)
        thresh = e_ticks * max(e_share, 1.0 - e_share)
        if run >= max(thresh, 2.0):
            starts.append(start)
            start = t + 1
    bounds = np.asarray([st for st in starts if st < p.size - 1], dtype=np.intp)
    if bounds.size == 0:
        raise ValueError("no complete run bars")
    return _ohlcv(p, s, bounds)


def dollar_imbalance_bars(
    prices: Array, sizes: Array, expected_ticks: int = 50, alpha_ewma: float = 0.1
) -> dict[str, Array]:
    """Dollar imbalance bars: close when ``|sum(b_t * dollar_t)|`` exceeds the
    EWMA-scaled threshold (AFML 2.5.2, dollar variant)."""
    p, s = _as_ticks(prices, sizes)
    if expected_ticks < 2 or not np.isfinite(alpha_ewma) or not (0.0 < alpha_ewma <= 1.0):
        raise ValueError("expected_ticks >= 2 and alpha_ewma in (0,1] required")
    b = _tick_signs(p)
    dollar = p * s * b
    starts: list[int] = []
    start = 0
    theta = 0.0
    e_imb = float(np.abs(dollar).mean())
    e_ticks = float(expected_ticks)
    for t in range(p.size):
        theta += dollar[t]
        e_imb = (1.0 - alpha_ewma) * e_imb + alpha_ewma * abs(dollar[t])
        if abs(theta) >= e_ticks * max(e_imb, 1e-12):
            starts.append(start)
            start = t + 1
            theta = 0.0
    bounds = np.asarray([st for st in starts if st < p.size - 1], dtype=np.intp)
    if bounds.size == 0:
        raise ValueError("no complete dollar-imbalance bars")
    return _ohlcv(p, s, bounds)
