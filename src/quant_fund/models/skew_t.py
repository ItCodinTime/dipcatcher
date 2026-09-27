"""Hansen (1994) skewed Student-t distribution.

Hansen's standardised skew-t has zero mean, unit variance, degrees of freedom
``nu > 2`` and skewness ``lambda in (-1, 1)``.  With

    c = Gamma((nu+1)/2) / (sqrt(pi (nu-2)) Gamma(nu/2)),
    a = 4 lambda c (nu-2)/(nu-1),   b = sqrt(1 + 3 lambda^2 - a^2),

the density of the standardised variable ``z`` is

    f(z) = b c (1 + (1/(nu-2)) ((b z + a)/(1 -/+ lambda))^2)^{-(nu+1)/2},

using ``1 - lambda`` for ``z < -a/b`` and ``1 + lambda`` otherwise.  A
location-scale version ``x = mu + sigma z`` is fitted by maximum likelihood.

Reference: B. E. Hansen (1994), "Autoregressive conditional density
estimation", International Economic Review.  Fail-closed on invalid parameters
or non-finite input.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, minimize
from scipy.special import gammaln
from scipy.stats import t as student_t

Array = NDArray[np.float64]


def _abc(nu: float, lam: float) -> tuple[float, float, float]:
    if nu <= 2.0 or not -1.0 < lam < 1.0:
        raise ValueError("require nu > 2 and lambda in (-1, 1)")
    log_c = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(np.pi * (nu - 2.0))
    c = float(np.exp(log_c))
    a = 4.0 * lam * c * (nu - 2.0) / (nu - 1.0)
    b = float(np.sqrt(1.0 + 3.0 * lam**2 - a**2))
    return a, b, c


def skew_t_pdf(x: Array, nu: float, lam: float, mu: float = 0.0, sigma: float = 1.0) -> Array:
    """Hansen skew-t density (location-scale)."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    a, b, c = _abc(nu, lam)
    z = (np.asarray(x, dtype=float) - mu) / sigma
    denom = np.where(z < -a / b, 1.0 - lam, 1.0 + lam)
    kernel = 1.0 + (1.0 / (nu - 2.0)) * ((b * z + a) / denom) ** 2
    return b * c * kernel ** (-(nu + 1.0) / 2.0) / sigma


def skew_t_cdf(x: Array, nu: float, lam: float, mu: float = 0.0, sigma: float = 1.0) -> Array:
    """Hansen skew-t CDF via the standard Student-t CDF of each spliced branch."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    a, b, c = _abc(nu, lam)  # noqa: F841 - c retained for symmetry/clarity
    z = (np.asarray(x, dtype=float) - mu) / sigma
    scale = np.sqrt(nu / (nu - 2.0))
    lo = (1.0 - lam) * student_t.cdf(scale * (b * z + a) / (1.0 - lam), nu)
    hi = (1.0 - lam) * 0.5 + (1.0 + lam) * (
        student_t.cdf(scale * (b * z + a) / (1.0 + lam), nu) - 0.5
    )
    return np.asarray(np.where(z < -a / b, lo, hi), dtype=float)


def skew_t_ppf(p: float, nu: float, lam: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Hansen skew-t quantile by inverting the CDF."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")

    def obj(x: float) -> float:
        return float(skew_t_cdf(np.array([x]), nu, lam, mu, sigma)[0] - p)

    return float(brentq(obj, mu - 200.0 * sigma, mu + 200.0 * sigma, xtol=1e-10))


def skew_t_fit(x: Array) -> dict[str, float]:
    """Maximum-likelihood fit returning ``nu``, ``lambda``, ``mu``, ``sigma``."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < 30 or not np.isfinite(arr).all():
        raise ValueError("x must be finite with >= 30 observations")
    mu0, sigma0 = float(arr.mean()), float(arr.std(ddof=1))
    if sigma0 <= 0.0:
        raise ValueError("data has zero variance")

    def nll(theta: Array) -> float:
        nu, lam, mu, log_sigma = theta
        if nu <= 2.01 or not -0.98 < lam < 0.98:
            return 1e12
        pdf = skew_t_pdf(arr, float(nu), float(lam), float(mu), float(np.exp(log_sigma)))
        if not np.all(pdf > 0) or not np.all(np.isfinite(pdf)):
            return 1e12
        return -float(np.log(pdf).sum())

    x0 = np.array([8.0, 0.0, mu0, np.log(sigma0)])
    bounds = [
        (2.05, 200.0),
        (-0.97, 0.97),
        (mu0 - 5 * sigma0, mu0 + 5 * sigma0),
        (np.log(sigma0) - 3, np.log(sigma0) + 3),
    ]
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    nu, lam, mu, log_sigma = res.x
    return {
        "nu": float(nu),
        "lam": float(lam),
        "mu": float(mu),
        "sigma": float(np.exp(log_sigma)),
        "loglik": float(-res.fun),
    }
