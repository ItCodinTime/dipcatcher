"""Generalized additive model via backfitting (Hastie & Tibshirani 1986).

An additive model ``y = intercept + sum_j f_j(x_j) + e`` is fitted by the
backfitting algorithm: cycle through the features, and re-estimate each smooth
``f_j`` from the partial residual ``y - intercept - sum_{k != j} f_k`` with a
one-dimensional smoother, centring each smooth for identifiability.  Here the
smoother is a Gaussian local-linear regression with a Silverman bandwidth.

Reference: T. Hastie, R. Tibshirani (1986), "Generalized additive models",
Statistical Science.  Fail-closed on non-finite input or degenerate features.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _silverman(x: Array) -> float:
    n = x.size
    sd = float(np.std(x))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    spread = min(sd, iqr / 1.34) if iqr > 0 else sd
    return max(0.9 * spread * n ** (-0.2), 1e-6)


def _local_linear(x: Array, y: Array, h: float) -> Array:
    """Local-linear smoother evaluated at the training points ``x``."""
    n = x.size
    out = np.empty(n)
    for i in range(n):
        w = np.exp(-0.5 * ((x - x[i]) / h) ** 2)
        sw = w.sum()
        mx = np.sum(w * x) / sw
        my = np.sum(w * y) / sw
        sxx = np.sum(w * (x - mx) ** 2)
        sxy = np.sum(w * (x - mx) * (y - my))
        beta = sxy / sxx if sxx > 0 else 0.0
        out[i] = my + beta * (x[i] - mx)
    return out


def gam_fit(x: Array, y: Array, max_iter: int = 25, tol: float = 1e-5) -> dict[str, Array | float]:
    """Backfitting GAM fit; returns intercept, per-feature contributions and R^2."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float).ravel()
    if xa.ndim == 1:
        xa = xa[:, None]
    if xa.shape[0] != ya.size or xa.shape[0] < 20 or not np.isfinite(xa).all():
        raise ValueError("x (n, p) and y must be finite, aligned, n >= 20")
    n, p = xa.shape
    intercept = float(ya.mean())
    contrib = np.zeros((n, p))
    bandwidths = [_silverman(xa[:, j]) for j in range(p)]
    for _ in range(max_iter):
        max_change = 0.0
        for j in range(p):
            partial = ya - intercept - (contrib.sum(axis=1) - contrib[:, j])
            f_new = _local_linear(xa[:, j], partial, bandwidths[j])
            f_new -= f_new.mean()  # centring for identifiability
            max_change = max(max_change, float(np.max(np.abs(f_new - contrib[:, j]))))
            contrib[:, j] = f_new
        if max_change < tol:
            break
    fitted = intercept + contrib.sum(axis=1)
    resid = ya - fitted
    ss_tot = float(((ya - ya.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    return {"intercept": intercept, "contributions": contrib, "fitted": fitted, "r2": r2}
