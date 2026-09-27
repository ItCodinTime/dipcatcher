"""Functional data analysis: FPCA and function-on-scalar regression
(Ramsay & Silverman).

Curves are stored as an (n, G) matrix observed on a common grid.
- ``fpca``: dense FPCA via eigendecomposition of the discretized
  covariance (trapezoid-weighted inner product). Returns mean curve,
  eigenfunctions, FPC scores, eigenvalues, and fraction of variance
  explained (FVE).
- ``fos_regress``: function-on-scalar regression y_i(t) = x_i' beta(t)
  estimated by pointwise OLS at each grid point.
- ``fpca_predict``: reconstruct curves from a subset of components.

Fail-closed: fewer than 5 curves, fewer than 5 grid points, non-finite.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _check_curves(y: Array) -> Array:
    yy = np.asarray(y, dtype=float)
    if yy.ndim != 2 or yy.shape[0] < 5 or yy.shape[1] < 5:
        raise ValueError("Y must be (n, G) with n >= 5, G >= 5")
    if not np.isfinite(yy).all():
        raise ValueError("non-finite input")
    return yy


def _trapz_weights(g: int) -> Array:
    w = np.ones(g)
    w[0] = w[-1] = 0.5
    return w / (g - 1)


def fpca(y: Array, n_comp: int | None = None) -> dict[str, Array | float]:
    """Functional PCA on a dense common grid (unit-spaced by default).

    Eigenfunctions are L2-normalized w.r.t. the trapezoid inner product;
    scores are the inner products of centered curves with eigenfunctions.
    """
    yy = _check_curves(y)
    n, g = yy.shape
    w = _trapz_weights(g)
    mu = yy.mean(axis=0)
    yc = yy - mu
    # discretized covariance operator T f = int C(t,s) f(s) ds.
    # Symmetrize the weighted operator: M = W^{1/2} C W^{1/2} is
    # symmetric; eigenfunctions phi = W^{-1/2} psi are w-orthonormal.
    c_plain = yc.T @ yc / n
    wsq = np.sqrt(w)
    m = np.diag(wsq) @ c_plain @ np.diag(wsq)
    vals, psi = np.linalg.eigh(m)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = psi[:, order] / wsq[:, None]  # phi = W^{-1/2} psi
    scores = (yc * w[None, :]) @ vecs  # (n, G)
    if n_comp is not None:
        if not 1 <= n_comp <= g:
            raise ValueError("n_comp out of range")
        vals_k = vals[:n_comp]
        vecs_k = vecs[:, :n_comp]
        scores_k = scores[:, :n_comp]
    else:
        vals_k, vecs_k, scores_k = vals, vecs, scores
    total = float(vals.sum())
    fve = vals_k / max(total, 1e-14)
    return {
        "mean": mu,
        "eigfuncs": vecs_k,
        "scores": scores_k,
        "eigenvalues": vals_k,
        "fve": fve,
        "weights": w,
    }


def fpca_predict(fit: dict[str, Array | float], scores: Array | None = None) -> Array:
    """Reconstruct curves: mu(t) + sum_k score_ik phi_k(t)."""
    mu = np.asarray(fit["mean"], dtype=float)
    phi = np.asarray(fit["eigfuncs"], dtype=float)
    s = (
        np.asarray(fit["scores"], dtype=float)
        if scores is None
        else np.asarray(scores, dtype=float)
    )
    if s.ndim == 1:
        s = s[:, None]
    if s.shape[1] != phi.shape[1]:
        raise ValueError("score/component dimension mismatch")
    return np.asarray(mu[None, :] + s @ phi.T, dtype=float)


def fos_regress(y: Array, x: Array) -> dict[str, Array | float]:
    """Function-on-scalar regression y_i(t) = x_i' beta(t) + e_i(t).

    x may be (n,) (intercept added) or (n, k) with an intercept column
    included by the caller. Returns beta functions (k', G), fitted
    curves, and per-point R2.
    """
    yy = _check_curves(y)
    n, g = yy.shape
    xx = np.asarray(x, dtype=float)
    if xx.ndim == 1:
        xx = np.column_stack([np.ones(n), xx])
    if xx.shape[0] != n or xx.ndim != 2 or not np.isfinite(xx).all():
        raise ValueError("x must be finite (n, k) aligned with Y")
    if np.linalg.matrix_rank(xx) < xx.shape[1]:
        raise ValueError("x is rank deficient")
    pinv = np.linalg.pinv(xx)
    beta = pinv @ yy  # (k, G) - all grid points at once
    fitted = xx @ beta
    ss_res = ((yy - fitted) ** 2).sum(axis=0)
    ss_tot = ((yy - yy.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-14)
    return {
        "beta": beta,
        "fitted": fitted,
        "r2": r2,
        "resid": yy - fitted,
    }
