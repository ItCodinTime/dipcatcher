"""Markowitz (1952) mean-variance efficient frontier (analytic).

With expected returns ``mu`` and covariance ``Sigma`` define the efficient-set
constants ``A = 1'Sigma^{-1}1``, ``B = 1'Sigma^{-1}mu``, ``C = mu'Sigma^{-1}mu``
and ``D = AC - B^2``.  The global minimum-variance portfolio is
``Sigma^{-1}1 / A``; the minimum-variance portfolio achieving target return
``m`` is ``w = g + h m`` and its variance is ``(A m^2 - 2 B m + C)/D``.  With a
risk-free rate ``rf`` the tangency portfolio is
``Sigma^{-1}(mu - rf 1) / (1'Sigma^{-1}(mu - rf 1))``.

Reference: H. Markowitz (1952), "Portfolio selection", Journal of Finance.
Fail-closed on non-finite or singular inputs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _abc(mu: Array, cov: Array) -> tuple[Array, Array, float, float, float, float]:
    m = np.asarray(mu, dtype=float).ravel()
    s = np.asarray(cov, dtype=float)
    n = m.size
    if s.shape != (n, n) or n < 2 or not (np.isfinite(m).all() and np.isfinite(s).all()):
        raise ValueError("mu (n,) and cov (n, n) must be finite and conformable")
    try:
        inv = np.linalg.inv(s)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance is singular") from exc
    ones = np.ones(n)
    inv1 = inv @ ones
    invmu = inv @ m
    a = float(ones @ inv1)
    b = float(ones @ invmu)
    c = float(m @ invmu)
    d = a * c - b**2
    if abs(d) < 1e-14:
        raise ValueError("degenerate frontier (D ~ 0)")
    return inv1, invmu, a, b, c, d


def min_variance_portfolio(cov: Array) -> Array:
    """Global minimum-variance weights."""
    s = np.asarray(cov, dtype=float)
    n = s.shape[0]
    if s.ndim != 2 or s.shape[1] != n or n < 2 or not np.isfinite(s).all():
        raise ValueError("cov must be a finite (n, n) matrix, n >= 2")
    try:
        inv1 = np.linalg.solve(s, np.ones(n))
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance is singular") from exc
    a = float(np.ones(n) @ inv1)
    if abs(a) < 1e-14:
        raise ValueError("degenerate covariance")
    return inv1 / a


def frontier_weights(mu: Array, cov: Array, target: float) -> Array:
    """Minimum-variance weights achieving expected return ``target``."""
    inv1, invmu, a, b, c, d = _abc(mu, cov)
    g = (c * inv1 - b * invmu) / d
    h = (a * invmu - b * inv1) / d
    return g + h * target


def frontier_variance(mu: Array, cov: Array, target: float) -> float:
    """Minimum variance attainable at expected return ``target``."""
    _, _, a, b, c, d = _abc(mu, cov)
    return float((a * target**2 - 2.0 * b * target + c) / d)


def tangency_portfolio(mu: Array, cov: Array, rf: float = 0.0) -> Array:
    """Tangency (maximum-Sharpe) portfolio for risk-free rate ``rf``."""
    m = np.asarray(mu, dtype=float).ravel()
    s = np.asarray(cov, dtype=float)
    _abc(m, s)  # validation
    excess = np.linalg.solve(s, m - rf)
    denom = float(np.sum(excess))
    if abs(denom) < 1e-14:
        raise ValueError("excess-return weights sum to zero; tangency undefined")
    return excess / denom
