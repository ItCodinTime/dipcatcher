"""Fractional response models for y in [0, 1] (Papke & Wooldridge 1996).

Quasi-Bernoulli likelihood: E[y|x] = G(x'b) with logit or probit G;
ll = sum [ y ln G + (1 - y) ln (1 - G) ]. The QMLE is consistent for
the conditional mean even when y is a genuine fraction (not 0/1).

Returns coefs, Hessian-based SEs, fitted means, quasi-loglik, and the
fractional R2 = squared correlation between y and fitted means.

Fail-closed: y outside [0, 1], non-finite X, rank-deficient X.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats

Array = NDArray[np.float64]


def _g(z: Array, link: str) -> Array:
    return stats.norm.cdf(z) if link == "probit" else 1.0 / (1.0 + np.exp(-z))


def _check(y: Array, x: Array) -> tuple[Array, Array]:
    yy = np.asarray(y, dtype=float).ravel()
    xx = np.asarray(x, dtype=float)
    if xx.ndim != 2 or xx.shape[0] != yy.size or yy.size < 30:
        raise ValueError("x must be (n, k), n >= 30")
    if not np.isfinite(yy).all() or not np.isfinite(xx).all():
        raise ValueError("non-finite input")
    if (yy < 0).any() or (yy > 1).any():
        raise ValueError("y must lie in [0, 1]")
    if np.linalg.matrix_rank(xx) < xx.shape[1]:
        raise ValueError("x is rank deficient")
    return yy, xx


def _num_hess(nll: object, theta: Array) -> Array:
    fn = nll  # type: ignore[assignment]
    k = theta.size
    h = 1e-5 * np.maximum(1.0, np.abs(theta))
    hh = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k)
            ej = np.zeros(k)
            ei[i] = h[i]
            ej[j] = h[j]
            hh[i, j] = hh[j, i] = (
                float(fn(theta + ei + ej))  # type: ignore[operator]
                - float(fn(theta + ei - ej))  # type: ignore[operator]
                - float(fn(theta - ei + ej))  # type: ignore[operator]
                + float(fn(theta - ei - ej))  # type: ignore[operator]
            ) / (4.0 * h[i] * h[j])
    return hh


def fractional_fit(y: Array, x: Array, link: str = "logit") -> dict[str, Array | float]:
    """Papke-Wooldridge fractional response QMLE."""
    yy, xx = _check(y, x)
    if link not in ("logit", "probit"):
        raise ValueError("link must be logit|probit")

    def nll(b: Array) -> float:
        xb = np.clip(xx @ b, -30, 30)
        g = np.clip(_g(xb, link), 1e-10, 1.0 - 1e-10)
        return float(-np.sum(yy * np.log(g) + (1 - yy) * np.log(1 - g)))

    res = optimize.minimize(nll, np.zeros(xx.shape[1]), method="BFGS")
    mu = _g(xx @ res.x, link)
    try:
        cov = np.linalg.inv(_num_hess(nll, res.x))
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        se = np.full(res.x.size, np.nan)
    return {
        "coef": res.x,
        "se": se,
        "fitted": mu,
        "loglik": float(-res.fun),
        "r2": float(np.corrcoef(yy, mu)[0, 1] ** 2),
        "link": float(link == "probit"),
    }
