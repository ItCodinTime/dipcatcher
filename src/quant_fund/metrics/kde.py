"""Gaussian kernel density estimation with data-driven bandwidths.

The Gaussian KDE is ``fhat(x) = (1/n h) sum_i phi((x - x_i)/h)``.  Three
bandwidth selectors are provided:

- Silverman's rule of thumb: ``h = 0.9 min(std, IQR/1.34) n^{-1/5}`` (Silverman
  1986).
- Scott's normal-reference rule: ``h = 1.059 std n^{-1/5}`` (Scott 1992).
- Least-squares (unbiased) cross-validation, minimising the integrated squared
  error estimate over ``h`` (Rudemo 1982; Bowman 1984); for the Gaussian kernel
  the criterion has a closed form.

Fail-closed on non-finite input or too little data.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

Array = NDArray[np.float64]

_INV_2SQRTPI = 1.0 / (2.0 * np.sqrt(np.pi))


def _as_vec(x: Array, min_obs: int = 10) -> Array:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < min_obs or not np.isfinite(arr).all():
        raise ValueError(f"x must be finite with >= {min_obs} observations")
    return arr


def silverman_bandwidth(x: Array) -> float:
    """Silverman (1986) rule-of-thumb bandwidth."""
    arr = _as_vec(x)
    n = arr.size
    iqr = float(np.subtract(*np.percentile(arr, [75, 25])))
    spread = min(float(arr.std(ddof=1)), iqr / 1.34) if iqr > 0 else float(arr.std(ddof=1))
    if spread <= 0.0:
        raise ValueError("data has zero spread")
    return float(0.9 * spread * n ** (-1.0 / 5.0))


def scott_bandwidth(x: Array) -> float:
    """Scott (1992) normal-reference bandwidth."""
    arr = _as_vec(x)
    sd = float(arr.std(ddof=1))
    if sd <= 0.0:
        raise ValueError("data has zero spread")
    return float(1.059 * sd * arr.size ** (-1.0 / 5.0))


def gaussian_kde(x: Array, points: Array, bandwidth: float) -> Array:
    """Evaluate the Gaussian KDE with the given ``bandwidth`` at ``points``."""
    arr = _as_vec(x)
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")
    pts = np.atleast_1d(np.asarray(points, dtype=float))
    u = (pts[:, None] - arr[None, :]) / bandwidth
    return np.asarray(norm.pdf(u).mean(axis=1) / bandwidth, dtype=float)


def _lscv(arr: Array, h: float) -> float:
    n = arr.size
    d = (arr[:, None] - arr[None, :]) / h
    term1 = float(np.sum(_INV_2SQRTPI * np.exp(-(d**2) / 4.0))) / (n**2 * h)
    off = norm.pdf(d)
    np.fill_diagonal(off, 0.0)
    term2 = 2.0 / (n * (n - 1) * h) * float(off.sum())
    return term1 - term2


def lscv_bandwidth(x: Array, n_grid: int = 40) -> float:
    """Least-squares cross-validation bandwidth (Gaussian kernel)."""
    arr = _as_vec(x)
    h0 = silverman_bandwidth(arr)
    grid = np.linspace(0.25 * h0, 3.0 * h0, n_grid)
    scores = np.array([_lscv(arr, float(h)) for h in grid])
    return float(grid[int(np.argmin(scores))])
