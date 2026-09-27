"""Gram-Charlier Type A / Edgeworth density with skewness and kurtosis terms.

The Gram-Charlier A series corrects a Gaussian with the first non-Gaussian
cumulants via Hermite polynomials:

    f(x) = phi(z)/sigma * [1 + (S/6) He_3(z) + (K/24) He_4(z)],

    F(x) = Phi(z) - phi(z) [ (S/6) He_2(z) + (K/24) He_3(z) ],

with ``z = (x - mu)/sigma``, skewness ``S`` and excess kurtosis ``K``, and the
probabilists' Hermite polynomials ``He_2 = z^2-1``, ``He_3 = z^3-3z``,
``He_4 = z^4-6z^2+3``.  The expansion is exact to third/fourth order but can go
locally negative for large ``|S|``/``K`` (the classic Gram-Charlier limitation);
``gram_charlier_valid`` reports whether the density is non-negative on a grid.

References: Charlier (1905); Edgeworth (1905); Jarrow & Rudd (1982) for the
option-pricing application.  Fail-closed on non-positive scale.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

Array = NDArray[np.float64]


def _z(x: Array, mu: float, sigma: float) -> Array:
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    return (np.asarray(x, dtype=float) - mu) / sigma


def gram_charlier_pdf(
    x: Array, mu: float = 0.0, sigma: float = 1.0, skew: float = 0.0, exkurt: float = 0.0
) -> Array:
    """Gram-Charlier Type A density."""
    z = _z(x, mu, sigma)
    he3 = z**3 - 3.0 * z
    he4 = z**4 - 6.0 * z**2 + 3.0
    correction = 1.0 + (skew / 6.0) * he3 + (exkurt / 24.0) * he4
    return np.asarray(norm.pdf(z) / sigma * correction, dtype=float)


def gram_charlier_cdf(
    x: Array, mu: float = 0.0, sigma: float = 1.0, skew: float = 0.0, exkurt: float = 0.0
) -> Array:
    """Gram-Charlier Type A CDF."""
    z = _z(x, mu, sigma)
    he2 = z**2 - 1.0
    he3 = z**3 - 3.0 * z
    return np.asarray(
        norm.cdf(z) - norm.pdf(z) * ((skew / 6.0) * he2 + (exkurt / 24.0) * he3), dtype=float
    )


def gram_charlier_valid(skew: float, exkurt: float, *, span: float = 6.0, n: int = 400) -> bool:
    """True when the standardised density is non-negative on ``[-span, span]``."""
    grid = np.linspace(-span, span, n)
    return bool((gram_charlier_pdf(grid, 0.0, 1.0, skew, exkurt) >= -1e-12).all())


def gram_charlier_fit(x: Array) -> dict[str, float]:
    """Fit by matching the first four sample moments."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < 10 or not np.isfinite(arr).all():
        raise ValueError("x must be finite with >= 10 observations")
    mu = float(arr.mean())
    sigma = float(arr.std(ddof=1))
    if sigma <= 0.0:
        raise ValueError("data has zero variance")
    zc = (arr - mu) / sigma
    skew = float((zc**3).mean())
    exkurt = float((zc**4).mean() - 3.0)
    return {
        "mu": mu,
        "sigma": sigma,
        "skew": skew,
        "exkurt": exkurt,
        "valid": float(gram_charlier_valid(skew, exkurt)),
    }
