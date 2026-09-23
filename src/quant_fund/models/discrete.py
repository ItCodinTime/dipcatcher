"""Discrete/limited dependent variable models: probit, logit, Tobit.

Maximum likelihood with BFGS; standard errors from the numerical
Hessian of the negative log-likelihood at the optimum. McFadden (1973)
pseudo-R2 = 1 - ll_model / ll_null reported for each fit.

Fail-closed: y not binary (probit/logit), collinear or non-finite X,
degenerate classes, Tobit with no censored mass.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats

Array = NDArray[np.float64]


def _check_xy(y: Array, x: Array) -> tuple[Array, Array]:
    yy = np.asarray(y, dtype=float).ravel()
    xx = np.asarray(x, dtype=float)
    if xx.ndim != 2 or xx.shape[0] != yy.size:
        raise ValueError("x must be (n, k) aligned with y")
    if not np.isfinite(yy).all() or not np.isfinite(xx).all():
        raise ValueError("non-finite input")
    if np.linalg.matrix_rank(xx) < xx.shape[1]:
        raise ValueError("x is rank deficient")
    return yy, xx


def _num_hess(f: object, theta: Array) -> Array:
    fn = f  # type: ignore[assignment]
    k = theta.size
    h = 1e-5 * np.maximum(1.0, np.abs(theta))
    hh = np.zeros((k, k))
    f0 = float(fn(theta))  # type: ignore[operator]
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k)
            ej = np.zeros(k)
            ei[i] = h[i]
            ej[j] = h[j]
            fpp = float(fn(theta + ei + ej))  # type: ignore[operator]
            fpm = float(fn(theta + ei - ej))  # type: ignore[operator]
            fmp = float(fn(theta - ei + ej))  # type: ignore[operator]
            fmm = float(fn(theta - ei - ej))  # type: ignore[operator]
            hh[i, j] = hh[j, i] = (fpp - fpm - fmp + fmm) / (4.0 * h[i] * h[j])
        _ = f0
    return hh


def _se_from_hess(nll: object, theta: Array) -> Array:
    try:
        h = _num_hess(nll, theta)
        cov = np.linalg.inv(h)
        return np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        return np.full(theta.size, np.nan)


def probit_fit(y: Array, x: Array) -> dict[str, Array | float]:
    """Binary probit: P(y=1|x) = Phi(x'b)."""
    yy, xx = _check_xy(y, x)
    if set(np.unique(yy)) - {0.0, 1.0} or yy.min() == yy.max():
        raise ValueError("y must be binary with both classes present")

    def nll(b: Array) -> float:
        xb = xx @ b
        ll = yy * np.log(np.clip(stats.norm.cdf(xb), 1e-12, 1.0)) + (1 - yy) * np.log(
            np.clip(stats.norm.cdf(-xb), 1e-12, 1.0)
        )
        return float(-np.sum(ll))

    res = optimize.minimize(nll, np.zeros(xx.shape[1]), method="BFGS")
    se = _se_from_hess(nll, res.x)
    ll_m = float(-res.fun)
    p0 = yy.mean()
    ll_0 = float(np.sum(yy * np.log(p0) + (1 - yy) * np.log(1 - p0)))
    return {
        "coef": res.x,
        "se": se,
        "loglik": ll_m,
        "mcfadden_r2": 1.0 - ll_m / ll_0,
        "fitted": stats.norm.cdf(xx @ res.x),
    }


def logit_fit(y: Array, x: Array) -> dict[str, Array | float]:
    """Binary logit: P(y=1|x) = Lambda(x'b)."""
    yy, xx = _check_xy(y, x)
    if set(np.unique(yy)) - {0.0, 1.0} or yy.min() == yy.max():
        raise ValueError("y must be binary with both classes present")

    def nll(b: Array) -> float:
        xb = xx @ b
        ll = yy * xb - np.logaddexp(0.0, xb)
        return float(-np.sum(ll))

    res = optimize.minimize(nll, np.zeros(xx.shape[1]), method="BFGS")
    se = _se_from_hess(nll, res.x)
    ll_m = float(-res.fun)
    p0 = yy.mean()
    ll_0 = float(np.sum(yy * np.log(p0) + (1 - yy) * np.log(1 - p0)))
    fitted = 1.0 / (1.0 + np.exp(-(xx @ res.x)))
    return {
        "coef": res.x,
        "se": se,
        "loglik": ll_m,
        "mcfadden_r2": 1.0 - ll_m / ll_0,
        "fitted": fitted,
    }


def tobit_fit(y: Array, x: Array, censor_at: float = 0.0) -> dict[str, Array | float]:
    """Left-censored Tobit: y* = x'b + e, y = max(y*, censor_at)."""
    yy, xx = _check_xy(y, x)
    cens = yy <= censor_at
    if cens.mean() == 0.0 or cens.mean() == 1.0:
        raise ValueError("need a mix of censored and uncensored observations")
    k = xx.shape[1]
    s0 = float(np.std(yy[~cens]))

    def nll(theta: Array) -> float:
        b = theta[:k]
        sig = np.exp(theta[k])
        xb = xx @ b
        ll_u = np.sum(stats.norm.logpdf((yy[~cens] - xb[~cens]) / sig) - np.log(sig))
        ll_c = np.sum(stats.norm.logcdf((censor_at - xb[cens]) / sig))
        return float(-(ll_u + ll_c))

    theta0 = np.concatenate([np.zeros(k), [np.log(max(s0, 1e-3))]])
    res = optimize.minimize(nll, theta0, method="BFGS")
    b_hat, sig_hat = res.x[:k], float(np.exp(res.x[k]))
    return {
        "coef": b_hat,
        "sigma": sig_hat,
        "loglik": float(-res.fun),
        "censored_share": float(cens.mean()),
        "latent_fitted": xx @ b_hat,
    }
