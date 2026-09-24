"""Johnson SU distribution (Johnson 1949) with Slifker-Shapiro quantile fit.

The unbounded Johnson family maps a variable to a standard normal through

    z = gamma + delta * asinh((x - xi) / lambda),   delta > 0, lambda > 0,

so ``x = xi + lambda * sinh((z - gamma) / delta)``.  It spans a wide range of
skewness/kurtosis and is a flexible heavy-tailed model for returns.

Parameters are estimated with the Slifker & Shapiro (1980) percentile method,
which reads four symmetric quantiles at ``z = +/- z0`` and ``z = +/- 3 z0`` and
solves the SU relations in closed form.  Fail-closed when the sample fails the
SU shape criterion (``m*n / p^2 > 1``) or on non-finite input.

References: N. L. Johnson (1949), Biometrika; J. Slifker & S. Shapiro (1980),
Technometrics.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

Array = NDArray[np.float64]


def johnson_su_pdf(
    x: Array, gamma: float, delta: float, xi: float = 0.0, lam: float = 1.0
) -> Array:
    """Johnson SU density."""
    if delta <= 0.0 or lam <= 0.0:
        raise ValueError("delta and lambda must be positive")
    u = (np.asarray(x, dtype=float) - xi) / lam
    z = gamma + delta * np.arcsinh(u)
    return (delta / (lam * np.sqrt(2.0 * np.pi))) / np.sqrt(1.0 + u * u) * np.exp(-0.5 * z * z)


def johnson_su_cdf(
    x: Array, gamma: float, delta: float, xi: float = 0.0, lam: float = 1.0
) -> Array:
    """Johnson SU CDF."""
    if delta <= 0.0 or lam <= 0.0:
        raise ValueError("delta and lambda must be positive")
    u = (np.asarray(x, dtype=float) - xi) / lam
    return norm.cdf(gamma + delta * np.arcsinh(u))


def johnson_su_ppf(
    p: float, gamma: float, delta: float, xi: float = 0.0, lam: float = 1.0
) -> float:
    """Johnson SU quantile."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if delta <= 0.0 or lam <= 0.0:
        raise ValueError("delta and lambda must be positive")
    return float(xi + lam * np.sinh((norm.ppf(p) - gamma) / delta))


def johnson_su_fit(x: Array, z0: float = 0.524) -> dict[str, float]:
    """Slifker-Shapiro percentile fit; returns ``gamma``, ``delta``, ``xi``, ``lam``."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < 20 or not np.isfinite(arr).all():
        raise ValueError("x must be finite with >= 20 observations")
    if z0 <= 0.0:
        raise ValueError("z0 must be positive")
    probs = norm.cdf(np.array([-3.0 * z0, -z0, z0, 3.0 * z0]))
    x_a, x_b, x_c, x_d = (float(v) for v in np.quantile(arr, probs))
    m = x_d - x_c
    n = x_b - x_a
    p = x_c - x_b
    if p <= 0.0 or m <= 0.0 or n <= 0.0:
        raise ValueError("degenerate quantiles; cannot fit")
    mp, np_ = m / p, n / p
    ratio = mp * np_
    if ratio <= 1.0:
        raise ValueError("sample does not satisfy the SU criterion (m*n/p^2 > 1)")
    delta = 2.0 * z0 / np.arccosh(0.5 * (mp + np_))
    gamma = delta * np.arcsinh((np_ - mp) / (2.0 * np.sqrt(ratio - 1.0)))
    # Recover scale/location exactly from the standardised inner spread implied
    # by the estimated (gamma, delta): p = lam * [sinh((z0-g)/d) - sinh((-z0-g)/d)].
    s_hi = np.sinh((z0 - gamma) / delta)
    s_lo = np.sinh((-z0 - gamma) / delta)
    lam = p / (s_hi - s_lo)
    xi = 0.5 * (x_c + x_b) - 0.5 * lam * (s_hi + s_lo)
    return {"gamma": float(gamma), "delta": float(delta), "xi": float(xi), "lam": float(lam)}
