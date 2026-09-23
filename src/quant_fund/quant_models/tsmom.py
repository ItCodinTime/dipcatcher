"""Moskowitz–Ooi–Pedersen time-series momentum.

davidalmeida90/quant-models ``time-series-momentum``. Sign of skipped
lookback excess return, scaled to a target vol. Research overlay, not a
live book and not a CS-ranker champion.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


def tsmom_weights(
    excess_returns: ArrayLike,
    *,
    lookback: int = 252,
    skip: int = 21,
    vol_lookback: int = 60,
    target_vol: float = 0.40,
    periods_per_year: int = 252,
) -> Array:
    """Weights for the last date. ``excess_returns`` is ``(T, N)`` simple excess.

    Signal is ``sign(sum_{t-lookback}^{t-skip} r)``. Size is
    ``target_vol / (sigma * sqrt(periods_per_year))``.
    """
    r = np.asarray(excess_returns, dtype=float)
    if r.ndim != 2:
        raise ValueError("excess_returns must be 2-D (dates, names)")
    t, n = r.shape
    if t < lookback + 1 or lookback <= skip:
        raise ValueError("need lookback > skip and enough rows")
    window = r[-(lookback):-skip] if skip > 0 else r[-lookback:]
    signal = np.sign(np.nansum(window, axis=0))
    vol_win = r[-vol_lookback:]
    sigma = np.nanstd(vol_win, axis=0, ddof=1)
    sigma = np.where(sigma > 1e-12, sigma, np.nan)
    size = target_vol / (sigma * np.sqrt(periods_per_year))
    w = signal * size
    w = np.where(np.isfinite(w), w, 0.0)
    return np.asarray(w, dtype=float)


def tsmom_panel(
    excess_returns: ArrayLike,
    **kwargs: float | int,
) -> Array:
    """Causal panel of TSMOM weights; row ``t`` uses data through ``t`` inclusive."""
    r = np.asarray(excess_returns, dtype=float)
    t, n = r.shape
    lookback = int(kwargs.get("lookback", 252))
    out = np.zeros((t, n), dtype=float)
    for i in range(lookback, t):
        out[i] = tsmom_weights(r[: i + 1], **kwargs)  # type: ignore[arg-type]
    return out
