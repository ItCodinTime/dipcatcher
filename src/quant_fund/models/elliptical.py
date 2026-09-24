"""Multivariate Student-t (elliptical) distribution fit via EM.

The multivariate t with location ``mu``, scatter ``Sigma`` and degrees of
freedom ``nu`` is the canonical heavy-tailed elliptical model.  It is fitted by
the EM algorithm (Liu & Rubin 1995): each iteration reweights observations by

    w_i = (nu + d) / (nu + (x_i - mu)' Sigma^{-1} (x_i - mu)),

then updates ``mu`` and ``Sigma`` as weighted moments.  ``nu`` is optionally
estimated by a 1-D search over the profile log-likelihood.

Reference: C. Liu, D. B. Rubin (1995), Statistica Sinica.  Fail-closed on
non-finite input or singular scatter.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

Array = NDArray[np.float64]


def _as_matrix(x: Array) -> Array:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < arr.shape[1] + 5 or not np.isfinite(arr).all():
        raise ValueError("x must be a finite (n, d) sample with n >= d + 5")
    return arr


def mv_t_logpdf(x: Array, mu: Array, sigma: Array, nu: float) -> Array:
    """Multivariate Student-t log density."""
    arr = np.atleast_2d(np.asarray(x, dtype=float))
    d = mu.size
    sign, logdet = np.linalg.slogdet(sigma)
    if sign <= 0:
        raise ValueError("sigma must be positive definite")
    inv = np.linalg.inv(sigma)
    diff = arr - mu
    maha = np.einsum("ij,jk,ik->i", diff, inv, diff)
    const = (
        gammaln((nu + d) / 2.0) - gammaln(nu / 2.0) - 0.5 * d * np.log(nu * np.pi) - 0.5 * logdet
    )
    return const - 0.5 * (nu + d) * np.log1p(maha / nu)


def mv_t_fit(
    x: Array, nu: float | None = None, max_iter: int = 200, tol: float = 1e-6
) -> dict[str, Array | float]:
    """EM fit of the multivariate t; estimates ``nu`` when not supplied."""
    arr = _as_matrix(x)
    n, d = arr.shape
    mu = arr.mean(axis=0)
    sigma = np.cov(arr, rowvar=False)
    estimate_nu = nu is None
    nu_val = 8.0 if nu is None else float(nu)
    if nu_val <= 2.0:
        raise ValueError("nu must be > 2")
    for _ in range(max_iter):
        inv = np.linalg.inv(sigma)
        diff = arr - mu
        maha = np.einsum("ij,jk,ik->i", diff, inv, diff)
        w = (nu_val + d) / (nu_val + maha)
        mu_new = (w[:, None] * arr).sum(axis=0) / w.sum()
        diff_new = arr - mu_new
        sigma_new = (w[:, None] * diff_new).T @ diff_new / n
        if estimate_nu:

            def neg_ll(log_nu: float, m: Array = mu_new, s: Array = sigma_new) -> float:
                return -float(mv_t_logpdf(arr, m, s, float(np.exp(log_nu))).sum())

            res = minimize_scalar(neg_ll, bounds=(np.log(2.1), np.log(200.0)), method="bounded")
            nu_val = float(np.exp(res.x))
        shift = float(np.max(np.abs(mu_new - mu))) + float(np.max(np.abs(sigma_new - sigma)))
        mu, sigma = mu_new, sigma_new
        if shift < tol:
            break
    return {"mu": mu, "sigma": sigma, "nu": float(nu_val)}
