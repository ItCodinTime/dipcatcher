"""Count data models: Poisson, negative binomial (NB2), zero-inflated
Poisson -- all MLE with log links and covariates.

- Poisson:  y ~ Pois(exp(x'b));  ll = y x'b - exp(x'b) - ln y!
- NB2:      variance = mu + a mu^2; a = exp(log_a) estimated jointly
- ZIP:      P(y=0) = w + (1-w) e^{-mu}, P(y>0) = (1-w) Pois(y|mu),
            w logistic-modeled on a separate design z (or constant).

Returns coefs, Hessian-based SEs, overdispersion check, AIC, fitted
means. Fail-closed: non-integer or negative y, non-finite X, rank-
deficient designs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize
from scipy.special import gammaln

Array = NDArray[np.float64]


def _check(y: Array, x: Array) -> tuple[Array, Array]:
    yy = np.asarray(y, dtype=float).ravel()
    xx = np.asarray(x, dtype=float)
    if xx.ndim != 2 or xx.shape[0] != yy.size:
        raise ValueError("x must be (n, k) aligned with y")
    if yy.size < 30 or not np.isfinite(yy).all() or not np.isfinite(xx).all():
        raise ValueError("insufficient or non-finite data")
    if (yy < 0).any() or np.abs(yy - np.round(yy)).max() > 1e-9:
        raise ValueError("y must be nonnegative integers")
    if np.linalg.matrix_rank(xx) < xx.shape[1]:
        raise ValueError("x is rank deficient")
    return np.round(yy), xx


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


def _se(nll: object, theta: Array) -> Array:
    try:
        cov = np.linalg.inv(_num_hess(nll, theta))
        return np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        return np.full(theta.size, np.nan)


def poisson_fit(y: Array, x: Array) -> dict[str, Array | float]:
    yy, xx = _check(y, x)

    def nll(b: Array) -> float:
        mu = np.exp(np.clip(xx @ b, -20, 20))
        return float(-np.sum(yy * (xx @ b) - mu - gammaln(yy + 1)))

    res = optimize.minimize(nll, np.zeros(xx.shape[1]), method="BFGS")
    mu = np.exp(xx @ res.x)
    return {
        "coef": res.x,
        "se": _se(nll, res.x),
        "mu": mu,
        "loglik": float(-res.fun),
        "aic": float(2 * res.x.size - 2 * (-res.fun)),
        "dispersion": float(np.sum(((yy - mu) ** 2 - yy) / mu) / (yy.size - res.x.size)),
    }


def nb2_fit(y: Array, x: Array) -> dict[str, Array | float]:
    """Negative binomial (NB2): Var = mu + alpha mu^2."""
    yy, xx = _check(y, x)
    k = xx.shape[1]

    def nll(th: Array) -> float:
        b = th[:k]
        alpha = np.exp(th[k])
        mu = np.exp(np.clip(xx @ b, -20, 20))
        r = 1.0 / alpha
        ll = (
            gammaln(yy + r)
            - gammaln(r)
            - gammaln(yy + 1)
            + r * np.log(r / (r + mu))
            + yy * np.log(mu / (r + mu))
        )
        return float(-np.sum(ll))

    th0 = np.concatenate([np.zeros(k), [0.0]])
    res = optimize.minimize(nll, th0, method="BFGS")
    b = res.x[:k]
    alpha = float(np.exp(res.x[k]))
    mu = np.exp(xx @ b)
    return {
        "coef": b,
        "alpha": alpha,
        "se": _se(nll, res.x)[:k],
        "mu": mu,
        "loglik": float(-res.fun),
        "aic": float(2 * (k + 1) - 2 * (-res.fun)),
    }


def zip_fit(y: Array, x: Array) -> dict[str, Array | float]:
    """Zero-inflated Poisson with constant inflation probability w."""
    yy, xx = _check(y, x)
    k = xx.shape[1]
    is0 = yy == 0

    def nll(th: Array) -> float:
        b = th[:k]
        logit_w = th[k]
        w = 1.0 / (1.0 + np.exp(-logit_w))
        mu = np.exp(np.clip(xx @ b, -20, 20))
        ll0 = np.log(w + (1.0 - w) * np.exp(-mu[is0]))
        llp = np.log(1.0 - w) + yy[~is0] * (xx[~is0] @ b) - mu[~is0] - gammaln(yy[~is0] + 1)
        return float(-(np.sum(ll0) + np.sum(llp)))

    th0 = np.concatenate([np.zeros(k), [0.0]])
    res = optimize.minimize(nll, th0, method="BFGS")
    b = res.x[:k]
    w = float(1.0 / (1.0 + np.exp(-res.x[k])))
    mu = np.exp(xx @ b)
    return {
        "coef": b,
        "w": w,
        "se": _se(nll, res.x)[:k],
        "mu": mu,
        "loglik": float(-res.fun),
        "aic": float(2 * (k + 1) - 2 * (-res.fun)),
        "zero_share": float(is0.mean()),
    }
