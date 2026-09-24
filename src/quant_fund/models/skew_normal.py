"""Azzalini (1985) skew-normal distribution.

The skew-normal SN(xi, omega, alpha) has density

    f(x) = (2 / omega) * phi(z) * Phi(alpha z),   z = (x - xi) / omega,

with location ``xi``, scale ``omega > 0`` and shape ``alpha`` (``alpha = 0``
recovers the normal).  The CDF uses Owen's T function,
``F(x) = Phi(z) - 2 T(z, alpha)``.

Method-of-moments fitting inverts the closed-form skewness relation (the
attainable sample skewness is bounded by |g1| < 0.9953).

Reference: A. Azzalini (1985), "A class of distributions which includes the
normal ones", Scand. J. Statist.  Fail-closed on non-finite input or
non-positive scale.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import owens_t
from scipy.stats import norm

Array = NDArray[np.float64]

_B = np.sqrt(2.0 / np.pi)
_MAX_SKEW = 0.995  # attainable |skewness| upper bound for the skew-normal


def _delta(alpha: float) -> float:
    return alpha / np.sqrt(1.0 + alpha * alpha)


def skew_normal_pdf(x: Array, xi: float = 0.0, omega: float = 1.0, alpha: float = 0.0) -> Array:
    """Skew-normal density."""
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    z = (np.asarray(x, dtype=float) - xi) / omega
    return (2.0 / omega) * norm.pdf(z) * norm.cdf(alpha * z)


def skew_normal_cdf(x: Array, xi: float = 0.0, omega: float = 1.0, alpha: float = 0.0) -> Array:
    """Skew-normal CDF via Owen's T function."""
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    z = (np.asarray(x, dtype=float) - xi) / omega
    return norm.cdf(z) - 2.0 * owens_t(z, alpha)


def skew_normal_ppf(p: float, xi: float = 0.0, omega: float = 1.0, alpha: float = 0.0) -> float:
    """Skew-normal quantile via bracketed root finding on the CDF."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if omega <= 0.0:
        raise ValueError("omega must be positive")

    def obj(x: float) -> float:
        return float(skew_normal_cdf(np.array([x]), xi, omega, alpha)[0] - p)

    lo, hi = xi - 40.0 * omega, xi + 40.0 * omega
    return float(brentq(obj, lo, hi, xtol=1e-10))


def skew_normal_fit(x: Array) -> dict[str, float]:
    """Method-of-moments fit returning ``xi``, ``omega``, ``alpha``."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < 10 or not np.isfinite(arr).all():
        raise ValueError("x must be finite with >= 10 observations")
    mean = float(arr.mean())
    var = float(arr.var(ddof=1))
    m3 = float(((arr - mean) ** 3).mean())
    g1 = m3 / (var**1.5) if var > 0 else 0.0
    g1 = float(np.clip(g1, -_MAX_SKEW, _MAX_SKEW))
    abs_g = abs(g1)
    c = (2.0 * abs_g / (4.0 - np.pi)) ** (1.0 / 3.0)
    delta_abs = np.sqrt((np.pi / 2.0) * c**2 / (1.0 + c**2))
    delta = float(np.sign(g1) * delta_abs) if g1 != 0 else 0.0
    delta = float(np.clip(delta, -0.9999, 0.9999))
    alpha = delta / np.sqrt(max(1.0 - delta * delta, 1e-12))
    omega = np.sqrt(var / max(1.0 - _B * _B * delta * delta, 1e-12))
    xi = mean - omega * _B * delta
    return {"xi": float(xi), "omega": float(omega), "alpha": float(alpha)}
