"""Robust location and slope estimators.

- Hodges & Lehmann (1963): the one-sample estimator is the median of the Walsh
  averages ``(x_i + x_j)/2`` (``i <= j``); the two-sample shift estimator is the
  median of pairwise differences ``x_i - y_j``.  Both are highly robust
  (breakdown ~29%) and consistent with the Wilcoxon signed-rank / Mann-Whitney
  statistics.
- Siegel (1982) repeated median: a slope estimator with 50% breakdown,
  ``slope = median_i median_{j != i} (y_j - y_i) / (x_j - x_i)`` and an
  analogous repeated-median intercept.

Fail-closed on non-finite input or too little data.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _vec(x: Array, name: str, min_obs: int = 2) -> Array:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < min_obs or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite with >= {min_obs} observations")
    return arr


def hodges_lehmann(x: Array) -> dict[str, float]:
    """One-sample Hodges-Lehmann location: median of Walsh averages."""
    arr = _vec(x, "x")
    iu, ju = np.triu_indices(arr.size)  # includes the diagonal (i == j)
    walsh = (arr[iu] + arr[ju]) / 2.0
    return {"estimate": float(np.median(walsh)), "n": float(arr.size)}


def hodges_lehmann_shift(x: Array, y: Array) -> dict[str, float]:
    """Two-sample Hodges-Lehmann shift: median of all ``x_i - y_j``."""
    xa = _vec(x, "x")
    ya = _vec(y, "y")
    diffs = xa[:, None] - ya[None, :]
    return {"estimate": float(np.median(diffs)), "n_x": float(xa.size), "n_y": float(ya.size)}


def siegel_repeated_median(x: Array, y: Array) -> dict[str, float]:
    """Siegel (1982) repeated-median linear fit (50% breakdown)."""
    xa = _vec(x, "x", min_obs=3)
    ya = _vec(y, "y", min_obs=3)
    if xa.size != ya.size:
        raise ValueError("x and y must share length")
    n = xa.size
    inner_slopes = np.empty(n)
    for i in range(n):
        dx = np.delete(xa, i) - xa[i]
        dy = np.delete(ya, i) - ya[i]
        ok = dx != 0.0
        if not ok.any():
            inner_slopes[i] = np.nan
            continue
        inner_slopes[i] = float(np.median(dy[ok] / dx[ok]))
    valid = np.isfinite(inner_slopes)
    if not valid.any():
        raise ValueError("no distinct x values to define a slope")
    slope = float(np.median(inner_slopes[valid]))
    intercept = float(np.median(ya - slope * xa))
    return {"slope": slope, "intercept": intercept, "n": float(n)}
