"""Expectile estimation and asymmetric-least-squares regression.

Newey & Powell (1987) introduced expectiles as the minimisers of an asymmetric
squared loss.  The ``tau``-expectile ``mu`` of a variable solves

    mu = argmin  E[ w_tau(y - mu) (y - mu)^2 ],   w_tau(u) = tau if u > 0 else 1 - tau.

Expectiles are the unique law-invariant, coherent, *elicitable* risk measures
(Bellini & Di Bernardino 2017): for ``tau >= 0.5`` the ``tau``-expectile of the
loss distribution is a coherent risk measure (the Expectile Value-at-Risk,
EVaR).  Estimation uses iteratively reweighted least squares, which converges
because the objective is convex.

Fail-closed on non-finite input, ``tau`` outside ``(0, 1)``, or rank-deficient
designs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _check_tau(tau: float) -> None:
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must be in (0, 1)")


def expectile(x: Array, tau: float = 0.5, tol: float = 1e-10, max_iter: int = 200) -> float:
    """Scalar ``tau``-expectile via iteratively reweighted averaging."""
    _check_tau(tau)
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < 2 or not np.isfinite(arr).all():
        raise ValueError("x must be finite with >= 2 observations")
    mu = float(arr.mean())
    for _ in range(max_iter):
        w = np.where(arr > mu, tau, 1.0 - tau)
        new_mu = float((w * arr).sum() / w.sum())
        if abs(new_mu - mu) <= tol * (1.0 + abs(mu)):
            mu = new_mu
            break
        mu = new_mu
    return mu


def expectile_regression(
    x: Array, y: Array, tau: float = 0.5, add_const: bool = True, max_iter: int = 200
) -> dict[str, Array | float]:
    """Asymmetric-least-squares (expectile) regression via IRLS."""
    _check_tau(tau)
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float).ravel()
    if xa.ndim == 1:
        xa = xa[:, None]
    if xa.shape[0] != ya.size or not np.isfinite(xa).all() or not np.isfinite(ya).all():
        raise ValueError("x and y must be finite and aligned")
    design = np.column_stack([np.ones(ya.size), xa]) if add_const else xa
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("design matrix is rank deficient")
    beta = np.linalg.lstsq(design, ya, rcond=None)[0]
    for _ in range(max_iter):
        resid = ya - design @ beta
        w = np.where(resid > 0.0, tau, 1.0 - tau)
        wsqrt = np.sqrt(w)
        new_beta = np.linalg.lstsq(design * wsqrt[:, None], ya * wsqrt, rcond=None)[0]
        if np.max(np.abs(new_beta - beta)) <= 1e-10 * (1.0 + np.max(np.abs(beta))):
            beta = new_beta
            break
        beta = new_beta
    resid = ya - design @ beta
    return {
        "coef": beta,
        "fitted": design @ beta,
        "resid": resid,
        "tau": float(tau),
    }


def expectile_var(losses: Array, tau: float = 0.95) -> float:
    """Expectile Value-at-Risk (EVaR): the ``tau``-expectile of the loss series.

    Coherent for ``tau >= 0.5`` (Bellini & Di Bernardino 2017).  ``losses`` use
    the positive-is-loss convention.
    """
    if tau < 0.5:
        raise ValueError("EVaR is coherent only for tau >= 0.5")
    return expectile(losses, tau)
