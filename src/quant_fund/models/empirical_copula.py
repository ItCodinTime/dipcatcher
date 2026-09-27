"""Empirical (Deheuvels 1979) copula and copula-based dependence.

Given a sample ``X`` of shape ``(n, d)`` the pseudo-observations are the scaled
ranks ``U_{ij} = rank(X_{ij}) / (n + 1)``.  The empirical copula is

    C_n(u) = (1/n) sum_i prod_j 1{ U_{ij} <= u_j },

a consistent nonparametric estimator of the underlying copula (Deheuvels 1979).
It supports estimation of Spearman's rho, and empirical lower/upper tail
dependence coefficients.

Fail-closed on non-finite input or degenerate shapes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

Array = NDArray[np.float64]


def _as_matrix(x: Array) -> Array:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] < 10 or not np.isfinite(arr).all():
        raise ValueError("x must be a finite (n, d) sample with n >= 10")
    return arr


def pseudo_observations(x: Array) -> Array:
    """Scaled-rank pseudo-observations ``rank / (n + 1)`` in ``(0, 1)``."""
    arr = _as_matrix(x)
    n = arr.shape[0]
    return np.column_stack([rankdata(arr[:, j]) / (n + 1.0) for j in range(arr.shape[1])])


def empirical_copula(u: Array, points: Array) -> Array:
    """Evaluate the empirical copula ``C_n`` at each row of ``points``."""
    uu = np.asarray(u, dtype=float)
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if uu.ndim != 2 or pts.shape[1] != uu.shape[1]:
        raise ValueError("points must have the same dimension as u")
    if uu.shape[0] < 1:
        raise ValueError("u must be non-empty")
    out = np.empty(pts.shape[0])
    for i, p in enumerate(pts):
        out[i] = float(np.mean(np.all(uu <= p[None, :], axis=1)))
    return out


def spearman_rho(x: Array, y: Array) -> float:
    """Spearman's rho from pseudo-observations (copula-based)."""
    ux = pseudo_observations(x)[:, 0]
    uy = pseudo_observations(y)[:, 0]
    return float(np.corrcoef(ux, uy)[0, 1])


def tail_dependence(x: Array, y: Array, q: float = 0.95) -> dict[str, float]:
    """Empirical lower/upper tail-dependence coefficients at level ``q``."""
    if not 0.5 < q < 1.0:
        raise ValueError("q must be in (0.5, 1)")
    ux = pseudo_observations(x)[:, 0]
    uy = pseudo_observations(y)[:, 0]
    lower_t = 1.0 - q
    c_ll = float(np.mean((ux <= lower_t) & (uy <= lower_t)))
    lower = c_ll / lower_t
    c_uu = float(np.mean((ux > q) & (uy > q)))
    upper = c_uu / (1.0 - q)
    return {"lower": float(lower), "upper": float(upper), "q": float(q)}
