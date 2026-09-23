"""Fractional differentiation (AFML ch. 5; Hosking 1981).

Stationarity vs memory trade-off: integer differencing (d=1) erases memory;
fractional ``d in (0,1)`` keeps long-memory structure while reaching
stationarity.  Weights follow the binomial series
``w_k = -w_{k-1} (d - k + 1) / k``.

References:
- Hosking (1981). Fractional differencing. *Biometrika* 68.
- López de Prado (2018). *Advances in Financial Machine Learning*, ch. 5 —
  fixed-width window fracdiff (FFD) and the minimum-d ADF search.
- Granger, Joyeux (1980). An introduction to long-memory time series models.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def frac_weights(d: float, size: int) -> Array:
    """Expanding-window fractional-differentiation weights ``w_0..w_{size-1}``.

    ``w_0 = 1``; ``w_k = -w_{k-1} * (d - k + 1) / k``.  For integer ``d`` the
    series terminates at ``w_{d+1} = 0`` (exact binomial expansion).
    """
    if not np.isfinite(d):
        raise ValueError("d must be finite")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("size must be a positive integer")
    w = np.empty(size)
    w[0] = 1.0
    for k in range(1, size):
        w[k] = -w[k - 1] * (d - k + 1.0) / k
    return w


def frac_weights_ffd(d: float, threshold: float = 1e-5) -> Array:
    """Fixed-width weights truncated where ``|w_k| < threshold`` (AFML FFD).

    The window expands until the tail mass is negligible — bounded memory so
    the transform is a finite convolution, which keeps PIT causality cheap.
    """
    if not np.isfinite(d):
        raise ValueError("d must be finite")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be positive and finite")
    if d >= 1.0 and float(d).is_integer():
        return frac_weights(d, int(d) + 1)
    w = [1.0]
    wk = 1.0
    k = 1
    while k < 1_000_000:
        wk *= -(d - k + 1.0) / k
        if abs(wk) < threshold:
            break
        w.append(wk)
        k += 1
    return np.asarray(w)


def frac_diff(x: Array, d: float, threshold: float = 1e-5) -> Array:
    """FFD-transform a series: ``x_d[t] = sum_k w_k x[t-k]``.

    Output length equals input; the first ``len(weights) - 1`` entries are
    ``nan`` (insufficient history — honest truncation, never zero-padded).
    """
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size == 0:
        raise ValueError("x must be non-empty")
    if not np.isfinite(v).all():
        raise ValueError("x must contain only finite values")
    w = frac_weights_ffd(d, threshold)
    if w.size >= v.size:
        raise ValueError("weight window exceeds series length")
    out = np.full(v.size, np.nan)
    out[w.size - 1 :] = np.convolve(v, w, mode="valid")
    return out


def min_d_stationary(
    x: Array,
    d_grid: Array | None = None,
    *,
    alpha: float = 0.05,
    threshold: float = 1e-5,
) -> dict[str, float]:
    """Minimum ``d`` whose FFD series passes ADF at level ``alpha`` (AFML 5.5).

    Scans ``d_grid`` (default 0.0..1.0 step 0.05), returns the first ``d`` with
    ADF p-value below ``alpha`` along with the p-value and the weights used.
    Returns ``d = nan`` when no grid point reaches significance.
    """
    from arch.unitroot import ADF

    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size < 50:
        raise ValueError("x must have at least 50 observations")
    if not np.isfinite(v).all():
        raise ValueError("x must contain only finite values")
    grid = np.arange(0.0, 1.01, 0.05) if d_grid is None else np.asarray(d_grid, dtype=float)
    best_d = float("nan")
    best_p = float("nan")
    for d in grid:
        if d < 0.0 or d > 1.0:
            continue
        try:
            xd = frac_diff(v, float(d), threshold)
        except ValueError:
            continue  # weight window exceeds the series at this d/threshold
        xd = xd[np.isfinite(xd)]
        if xd.size < 30 or np.std(xd) <= 0.0:
            continue
        try:
            p = float(ADF(xd).pvalue)
        except Exception:  # ADF failure is a skip, not a crash
            continue
        if p < alpha:
            best_d, best_p = float(d), p
            break
    return {"d": best_d, "adf_pvalue": best_p, "alpha": float(alpha)}


def expanding_frac_diff(x: Array, d: float) -> Array:
    """Full expanding-window transform (unbounded weights; Hosking form).

    Slower than FFD (weights grow with t) but exact — useful for correctness
    checks of the truncated variant.  First entry is ``x[0]``.
    """
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size == 0:
        raise ValueError("x must be non-empty")
    if not np.isfinite(v).all():
        raise ValueError("x must contain only finite values")
    w = frac_weights(d, v.size)
    out = np.empty(v.size)
    for t in range(v.size):
        out[t] = float(np.dot(w[: t + 1][::-1], v[: t + 1]))
    return out
