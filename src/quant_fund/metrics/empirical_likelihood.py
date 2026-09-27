"""Owen (1988) empirical likelihood for the mean.

Empirical likelihood is a nonparametric likelihood placing weights ``p_i`` on
the observations.  For a hypothesised mean ``mu0`` the profile empirical
likelihood ratio statistic is

    W(mu0) = 2 sum_i log(1 + lambda (x_i - mu0)),

where the Lagrange multiplier ``lambda`` solves
``sum_i (x_i - mu0)/(1 + lambda (x_i - mu0)) = 0``.  Under H0 the mean equals
``mu0``, ``W`` is asymptotically chi-squared with one degree of freedom (Owen's
empirical-likelihood theorem), giving distribution-free tests and confidence
intervals.

References: A. Owen (1988), Biometrika; A. Owen (1990), Annals of Statistics.
Fail-closed on non-finite input or a ``mu0`` outside the data range.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.stats import chi2

Array = NDArray[np.float64]


def _as_vec(x: Array) -> Array:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < 5 or not np.isfinite(arr).all():
        raise ValueError("x must be finite with >= 5 observations")
    return arr


def _neg2_logelr(z: Array) -> float:
    """-2 log ELR for centred data ``z = x - mu0`` (mean-zero constraint)."""
    zmin, zmax = float(z.min()), float(z.max())
    if zmin >= 0.0 or zmax <= 0.0:
        return np.inf  # mu0 outside the convex hull -> empty feasible set
    n = z.size
    # lambda is bounded so that 1 + lambda z_i > 0 for all i.
    lo = (1.0 / n - 1.0) / zmax
    hi = (1.0 / n - 1.0) / zmin

    def estimating_eq(lam: float) -> float:
        return float(np.sum(z / (1.0 + lam * z)))

    lam = float(brentq(estimating_eq, lo, hi, xtol=1e-12))
    return 2.0 * float(np.sum(np.log1p(lam * z)))


def empirical_likelihood_mean(x: Array, mu0: float) -> dict[str, float]:
    """EL ratio test of ``H0: E[X] = mu0``; returns the statistic and p-value."""
    arr = _as_vec(x)
    if not (arr.min() < mu0 < arr.max()):
        raise ValueError("mu0 must lie strictly inside the data range")
    stat = _neg2_logelr(arr - mu0)
    return {"statistic": float(stat), "pvalue": float(chi2.sf(stat, 1)), "n": float(arr.size)}


def empirical_likelihood_ci(x: Array, level: float = 0.95) -> dict[str, float]:
    """EL confidence interval for the mean by inverting the ratio test."""
    arr = _as_vec(x)
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    crit = float(chi2.ppf(level, 1))
    xbar = float(arr.mean())
    lo_b, hi_b = float(arr.min()), float(arr.max())

    def excess(mu0: float) -> float:
        return _neg2_logelr(arr - mu0) - crit

    lower = float(brentq(excess, lo_b + 1e-9, xbar, xtol=1e-8))
    upper = float(brentq(excess, xbar, hi_b - 1e-9, xtol=1e-8))
    return {"lower": lower, "upper": upper, "mean": xbar, "level": float(level)}
