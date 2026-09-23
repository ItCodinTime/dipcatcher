"""Quantile autoregression (Koenker & Xiao 2006).

QAR(p): Q_{y_t}(tau | F_{t-1}) = a0(tau) + sum_j a_j(tau) y_{t-j}.
Each quantile is estimated by Koenker-Bassett quantile regression on
lagged values (delegated to ``metrics.regression.quantile_regression``).

Unit-root diagnostics across the quantile surface: the persistence
estimate a1(tau) may exceed 1 in the lower tail (local explosion)
even when the median process is stationary -- this is the asymmetric
adjustment phenomenon the QAR paper documents.

Fail-closed: tau outside (0,1), insufficient data, non-finite y.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.regression import quantile_regression

Array = NDArray[np.float64]


def _check(y: Array, p: int) -> Array:
    yy = np.asarray(y, dtype=float).ravel()
    if yy.size < 4 * p + 40 or not np.isfinite(yy).all():
        raise ValueError("insufficient or non-finite data")
    if p < 1:
        raise ValueError("p must be >= 1")
    return yy


def qar_fit(y: Array, p: int, taus: Array) -> dict[str, Array]:
    """Fit QAR(p) at each tau. Returns coef matrix (len(taus), p+1)."""
    yy = _check(y, p)
    tt = np.asarray(taus, dtype=float).ravel()
    if tt.size == 0 or (tt <= 0).any() or (tt >= 1).any():
        raise ValueError("taus must lie in (0, 1)")
    n = yy.size
    xl = np.column_stack([yy[p - i - 1 : n - i - 1] for i in range(p)])
    yt = yy[p:]
    coefs = np.empty((tt.size, p + 1))
    for i, tau in enumerate(tt):
        out = quantile_regression(xl, yt, float(tau))
        coefs[i] = np.asarray(out["beta"], dtype=float)
    x = np.column_stack([np.ones(n - p), xl])
    return {"coef": coefs, "taus": tt, "x": x, "y": yt}


def qar_summary(fit: dict[str, Array]) -> dict[str, Array]:
    """Persistence profile a1(tau) and a Kolmogorov-Smirnov-style
    uniformity diagnostic on the fitted quantile surface."""
    coef = np.asarray(fit["coef"], dtype=float)
    taus = np.asarray(fit["taus"], dtype=float)
    x = np.asarray(fit["x"], dtype=float)
    y = np.asarray(fit["y"], dtype=float)
    # fraction of realizations below their fitted conditional quantile
    # should equal tau on average
    below = np.zeros(taus.size)
    for i in range(taus.size):
        q = x @ coef[i]
        below[i] = float((y <= q).mean())
    return {
        "persistence_a1": coef[:, 1],
        "coverage": below,
        "coverage_dev": np.abs(below - taus),
    }
