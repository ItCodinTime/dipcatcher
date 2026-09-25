"""Feasible GLS for regression with AR(1) errors.

When the regression errors follow ``u_t = rho u_{t-1} + e_t`` OLS remains
unbiased but is inefficient and its standard errors are wrong.  Two classic
feasible-GLS estimators iterate between estimating ``rho`` from the residuals
and re-fitting the quasi-differenced regression:

- **Cochrane-Orcutt** (1949) quasi-differences ``y_t - rho y_{t-1}`` and drops
  the first observation.
- **Prais-Winsten** (1954) retains the first observation with the scaling
  ``sqrt(1 - rho^2)``, recovering full efficiency.

Fail-closed on non-finite input, rank-deficient designs, or too little data.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _prepare(y: Array, x: Array, add_const: bool) -> tuple[Array, Array]:
    ya = np.asarray(y, dtype=float).ravel()
    xa = np.asarray(x, dtype=float)
    if xa.ndim == 1:
        xa = xa[:, None]
    if xa.shape[0] != ya.size or not np.isfinite(ya).all() or not np.isfinite(xa).all():
        raise ValueError("y and x must be finite and aligned")
    if ya.size < xa.shape[1] + 10:
        raise ValueError("insufficient observations")
    design = np.column_stack([np.ones(ya.size), xa]) if add_const else xa
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("design matrix is rank deficient")
    return ya, design


def _ols(y: Array, x: Array) -> Array:
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _rho_from_resid(resid: Array) -> float:
    num = float(np.dot(resid[1:], resid[:-1]))
    den = float(np.dot(resid[:-1], resid[:-1]))
    return num / den if den > 0 else 0.0


def _se(y: Array, x: Array, beta: Array) -> Array:
    n, k = x.shape
    resid = y - x @ beta
    sigma2 = float(resid @ resid) / (n - k)
    cov = sigma2 * np.linalg.inv(x.T @ x)
    return np.sqrt(np.maximum(np.diag(cov), 0.0))


def cochrane_orcutt(
    y: Array, x: Array, add_const: bool = True, max_iter: int = 50, tol: float = 1e-6
) -> dict[str, Array | float]:
    """Cochrane-Orcutt (1949) iterative FGLS for AR(1) errors."""
    ya, design = _prepare(y, x, add_const)
    beta = _ols(ya, design)
    rho = _rho_from_resid(ya - design @ beta)
    iters = 0
    for _ in range(max_iter):
        iters += 1
        yt = ya[1:] - rho * ya[:-1]
        xt = design[1:] - rho * design[:-1]
        beta = _ols(yt, xt)
        new_rho = _rho_from_resid(ya - design @ beta)
        if abs(new_rho - rho) < tol:
            rho = new_rho
            break
        rho = new_rho
    yt = ya[1:] - rho * ya[:-1]
    xt = design[1:] - rho * design[:-1]
    return {"beta": beta, "rho": float(rho), "se": _se(yt, xt, beta), "iterations": float(iters)}


def prais_winsten(
    y: Array, x: Array, add_const: bool = True, max_iter: int = 50, tol: float = 1e-6
) -> dict[str, Array | float]:
    """Prais-Winsten (1954) FGLS retaining the first observation."""
    ya, design = _prepare(y, x, add_const)
    beta = _ols(ya, design)
    rho = _rho_from_resid(ya - design @ beta)
    iters = 0
    for _ in range(max_iter):
        iters += 1
        w = np.sqrt(max(1.0 - rho**2, 1e-12))
        yt = np.concatenate([[w * ya[0]], ya[1:] - rho * ya[:-1]])
        xt = np.vstack([w * design[0], design[1:] - rho * design[:-1]])
        beta = _ols(yt, xt)
        new_rho = _rho_from_resid(ya - design @ beta)
        if abs(new_rho - rho) < tol:
            rho = new_rho
            break
        rho = new_rho
    w = np.sqrt(max(1.0 - rho**2, 1e-12))
    yt = np.concatenate([[w * ya[0]], ya[1:] - rho * ya[:-1]])
    xt = np.vstack([w * design[0], design[1:] - rho * design[:-1]])
    return {"beta": beta, "rho": float(rho), "se": _se(yt, xt, beta), "iterations": float(iters)}
