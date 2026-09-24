"""Normal-Inverse-Gaussian and Variance-Gamma distributions.

Both are normal mean-variance mixtures widely used for asset returns:

- **NIG** (Barndorff-Nielsen 1997) mixes over an inverse-Gaussian subordinator.
  With tail ``alpha``, asymmetry ``beta`` (``|beta| < alpha``), scale
  ``delta > 0``, location ``mu`` and ``gamma = sqrt(alpha^2 - beta^2)``:

      f(x) = (alpha delta / pi) exp(delta gamma + beta (x-mu))
             K_1(alpha g) / g,   g = sqrt(delta^2 + (x-mu)^2).

- **Variance-Gamma** (Madan-Seneta 1990; Madan-Carr-Chang 1998) mixes over a
  gamma subordinator with scale ``sigma``, variance-rate ``nu`` and skew
  ``theta``.

Both provide a closed-form method-of-moments fit and an exact mixture sampler.
Densities use the modified Bessel function ``K`` (``scipy.special.kv``).
Fail-closed on invalid parameters or non-finite input.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import gamma as gamma_fn
from scipy.special import kv
from scipy.stats import invgauss

Array = NDArray[np.float64]


# --------------------------------------------------------------------------- NIG
def nig_pdf(x: Array, alpha: float, beta: float, delta: float, mu: float = 0.0) -> Array:
    """Normal-Inverse-Gaussian density."""
    if alpha <= 0.0 or delta <= 0.0 or abs(beta) >= alpha:
        raise ValueError("require alpha > 0, delta > 0, |beta| < alpha")
    g = np.sqrt(alpha**2 - beta**2)
    c = np.asarray(x, dtype=float) - mu
    rad = np.sqrt(delta**2 + c**2)
    return alpha * delta / np.pi * np.exp(delta * g + beta * c) * kv(1, alpha * rad) / rad


def nig_fit_moments(x: Array) -> dict[str, float]:
    """Closed-form method-of-moments NIG fit (alpha, beta, delta, mu)."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < 20 or not np.isfinite(arr).all():
        raise ValueError("x must be finite with >= 20 observations")
    m = float(arr.mean())
    v = float(arr.var(ddof=1))
    z = (arr - m) / np.sqrt(v)
    s = float((z**3).mean())
    k = float((z**4).mean() - 3.0)
    r = s**2 / k if k > 0 else 0.0
    if not 0.0 <= r < 0.75:
        raise ValueError("moments outside the NIG-admissible region (need k > 4 s^2/3)")
    bar_beta2 = r / (3.0 - 4.0 * r)
    bar_beta2 = float(min(bar_beta2, 0.999))
    bar_beta = float(np.sign(s) * np.sqrt(bar_beta2)) if s != 0 else 0.0
    one_minus = 1.0 - bar_beta2
    alpha = float(np.sqrt(3.0 * (1.0 + 4.0 * bar_beta2) / (k * v * one_minus**2)))
    beta = bar_beta * alpha
    g = alpha * np.sqrt(one_minus)
    delta = v * alpha * one_minus**1.5
    mu = m - delta * beta / g
    return {"alpha": alpha, "beta": float(beta), "delta": float(delta), "mu": float(mu)}


def nig_rvs(
    alpha: float,
    beta: float,
    delta: float,
    mu: float = 0.0,
    size: int = 1000,
    rng: np.random.Generator | None = None,
) -> Array:
    """Sample NIG via its inverse-Gaussian normal mean-variance mixture."""
    if alpha <= 0.0 or delta <= 0.0 or abs(beta) >= alpha:
        raise ValueError("require alpha > 0, delta > 0, |beta| < alpha")
    gen = np.random.default_rng() if rng is None else rng
    g = np.sqrt(alpha**2 - beta**2)
    zmean = delta / g
    z = invgauss.rvs(mu=zmean / delta**2, scale=delta**2, size=size, random_state=gen)
    return np.asarray(mu + beta * z + np.sqrt(z) * gen.standard_normal(size), dtype=float)


# ---------------------------------------------------------------- Variance-Gamma
def vg_pdf(x: Array, sigma: float, nu: float, theta: float = 0.0, mu: float = 0.0) -> Array:
    """Variance-Gamma density (Madan-Carr-Chang 1998)."""
    if sigma <= 0.0 or nu <= 0.0:
        raise ValueError("require sigma > 0 and nu > 0")
    c = np.asarray(x, dtype=float) - mu
    c = np.where(np.abs(c) < 1e-12, 1e-12, c)
    m_par = np.sqrt(theta**2 + 2.0 * sigma**2 / nu)
    order = 1.0 / nu - 0.5
    coef = (
        2.0
        * np.exp(theta * c / sigma**2)
        / (nu ** (1.0 / nu) * np.sqrt(2.0 * np.pi) * sigma * gamma_fn(1.0 / nu))
    )
    return coef * (np.abs(c) / m_par) ** order * kv(order, np.abs(c) * m_par / sigma**2)


def vg_fit_moments(x: Array) -> dict[str, float]:
    """Approximate method-of-moments Variance-Gamma fit (sigma, nu, theta, mu)."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < 20 or not np.isfinite(arr).all():
        raise ValueError("x must be finite with >= 20 observations")
    m = float(arr.mean())
    v = float(arr.var(ddof=1))
    zc = (arr - m) / np.sqrt(v)
    skew = float((zc**3).mean())
    exkurt = float((zc**4).mean() - 3.0)
    nu = float(np.clip(exkurt / 3.0, 1e-3, 10.0))
    sigma = float(np.sqrt(v))
    theta = float(skew * np.sqrt(v) / (3.0 * nu)) if nu > 0 else 0.0
    mu = m - theta
    return {"sigma": sigma, "nu": nu, "theta": theta, "mu": mu}


def vg_rvs(
    sigma: float,
    nu: float,
    theta: float = 0.0,
    mu: float = 0.0,
    size: int = 1000,
    rng: np.random.Generator | None = None,
) -> Array:
    """Sample VG via its gamma normal mean-variance mixture."""
    if sigma <= 0.0 or nu <= 0.0:
        raise ValueError("require sigma > 0 and nu > 0")
    gen = np.random.default_rng() if rng is None else rng
    g = gen.gamma(shape=1.0 / nu, scale=nu, size=size)  # mean 1, variance nu
    return np.asarray(mu + theta * g + sigma * np.sqrt(g) * gen.standard_normal(size), dtype=float)
