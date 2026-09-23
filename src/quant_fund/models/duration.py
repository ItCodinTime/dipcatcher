"""Engle-Russell (1998) Autoregressive Conditional Duration.

ACD(1,1): x_i = psi_i eps_i,  psi_i = omega + alpha x_{i-1} + beta psi_{i-1},
eps iid with E[eps]=1. EACD uses Exp(1) errors; WACD uses a mean-1
Weibull (Burr-free parameterization: eps ~ Weibull(gamma,
Gamma(1+1/gamma))).

QMLE via L-BFGS-B on (omega, alpha, beta[, gamma]) with stationarity
bounds alpha+beta<1, alpha,beta>=0, omega>0. Returns conditional
durations and standardized residuals x/psi for diagnostics.

Fail-closed: non-positive/non-finite durations, too few observations.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import optimize
from scipy.special import gammaln
from scipy.stats import chi2

Array = NDArray[np.float64]


def _psi_path(x: Array, omega: float, alpha: float, beta: float) -> Array:
    n = x.size
    psi = np.empty(n)
    psi[0] = max(float(x.mean()), 1e-9)
    for i in range(1, n):
        psi[i] = omega + alpha * x[i - 1] + beta * psi[i - 1]
        if psi[i] <= 1e-12:
            psi[i] = 1e-12
    return psi


def _nll_exp(theta: Array, x: Array) -> float:
    omega, alpha, beta = float(theta[0]), float(theta[1]), float(theta[2])
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return 1e12
    psi = _psi_path(x, omega, alpha, beta)
    return float(np.sum(np.log(psi) + x / psi))


def _nll_weibull(theta: Array, x: Array) -> float:
    omega, alpha, beta, gamma = (
        float(theta[0]),
        float(theta[1]),
        float(theta[2]),
        float(theta[3]),
    )
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999 or gamma <= 0.05 or gamma > 10:
        return 1e12
    psi = _psi_path(x, omega, alpha, beta)
    lam = math.exp(gammaln(1.0 + 1.0 / gamma))  # scale for mean-1 eps
    z = x / psi / lam
    # eps ~ Weibull(gamma, lam): f(e) = (g/lam)(e/lam)^{g-1} exp(-(e/lam)^g)
    # x = psi*eps -> loglik = sum[ ln g - ln psi - g ln lam + (g-1) ln x - (x/(psi lam))^g ]
    ll = np.sum(
        np.log(gamma)
        - np.log(psi)
        - gamma * np.log(lam)
        + (gamma - 1.0) * np.log(np.maximum(x, 1e-300))
        - z**gamma
    )
    return float(-ll)


def acd_fit(
    durations: Array,
    dist: str = "exp",
    init: Array | None = None,
) -> dict[str, Array | float]:
    """Fit EACD(1,1) or WACD(1,1) by MLE.

    ``durations``: positive inter-event times (already diurnally
    adjusted if needed — caller's responsibility). Returns params,
    psi path, standardized residuals, log-likelihood.
    """
    x = np.asarray(durations, dtype=float).ravel()
    if x.size < 50 or not np.isfinite(x).all() or (x <= 0).any():
        raise ValueError("durations must be finite, > 0, >= 50 obs")
    if dist not in ("exp", "weibull"):
        raise ValueError("dist must be 'exp' or 'weibull'")
    mu = float(x.mean())
    if init is None:
        theta0 = np.array([0.05 * mu, 0.05, 0.9])
        if dist == "weibull":
            theta0 = np.concatenate([theta0, [1.2]])
    else:
        theta0 = np.asarray(init, dtype=float).ravel()
    nll = _nll_exp if dist == "exp" else _nll_weibull
    bounds = [(1e-9, mu * 2.0), (0.0, 0.999), (0.0, 0.999)]
    if dist == "weibull":
        bounds.append((0.05, 10.0))
    res = optimize.minimize(nll, theta0, args=(x,), method="L-BFGS-B", bounds=bounds)
    if not np.isfinite(res.fun):
        raise ValueError("ACD fit failed")
    th = res.x
    omega, alpha, beta = float(th[0]), float(th[1]), float(th[2])
    psi = _psi_path(x, omega, alpha, beta)
    resid = x / psi
    out: dict[str, Array | float] = {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": alpha + beta,
        "psi": psi,
        "resid": resid,
        "loglik": float(-res.fun),
        "converged": float(res.success),
        "dist": float(dist == "weibull"),
    }
    if dist == "weibull":
        out["gamma"] = float(th[3])
    return out


def acd_simulate(
    omega: float,
    alpha: float,
    beta: float,
    n: int,
    rng: np.random.Generator,
    burn: int = 200,
) -> Array:
    """Simulate EACD(1,1) durations (Exp(1) errors)."""
    if not np.isfinite([omega, alpha, beta]).all():
        raise ValueError("params must be finite")
    if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or alpha + beta >= 1.0:
        raise ValueError("need omega>0, alpha,beta>=0, alpha+beta<1")
    if n < 10:
        raise ValueError("n >= 10")
    total = n + burn
    eps = rng.exponential(1.0, size=total)
    x = np.empty(total)
    psi = np.empty(total)
    psi[0] = omega / (1.0 - alpha - beta)
    x[0] = psi[0] * eps[0]
    for i in range(1, total):
        psi[i] = omega + alpha * x[i - 1] + beta * psi[i - 1]
        x[i] = psi[i] * eps[i]
    return np.asarray(x[burn:], dtype=float)


def acd_diagnostics(resid: Array, n_lags: int = 10) -> dict[str, float]:
    """Diagnostics on standardized residuals x/psi: should be ~iid mean-1.

    Returns mean, Ljung-Box stat/pvalue on levels, and excess kurtosis.
    """
    e = np.asarray(resid, dtype=float).ravel()
    if e.size < n_lags + 10 or not np.isfinite(e).all():
        raise ValueError("resid must be finite, > n_lags+10 obs")
    n = e.size
    ac = np.empty(n_lags)
    ec = e - e.mean()
    denom = float(ec @ ec)
    for lag in range(1, n_lags + 1):
        ac[lag - 1] = float(ec[lag:] @ ec[:-lag]) / denom if denom > 0 else 0.0
    lb = n * (n + 2.0) * float(np.sum(ac**2 / np.arange(n - 1, n - n_lags - 1, -1)))
    p = float(chi2.sf(lb, n_lags))
    m4 = float(np.mean((e - e.mean()) ** 4))
    kurt = m4 / max(float(e.var()) ** 2, 1e-18) - 3.0
    return {
        "mean_resid": float(e.mean()),
        "ljung_box": lb,
        "lb_pvalue": p,
        "excess_kurtosis": kurt,
    }
