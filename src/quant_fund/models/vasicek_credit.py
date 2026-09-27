"""Vasicek (2002) large-homogeneous-portfolio (ASRF) credit loss distribution.

In the single-factor asymptotic model each obligor defaults when a latent
variable ``sqrt(rho) M + sqrt(1-rho) Z`` falls below ``Phi^{-1}(pd)``.  For an
infinitely fine homogeneous pool the loss fraction equals the factor-conditional
default probability, giving the closed-form loss distribution

    P(L <= x) = Phi( (sqrt(1-rho) Phi^{-1}(x) - Phi^{-1}(pd)) / sqrt(rho) ),

with quantile (Basel IRB) ``L_q = Phi((Phi^{-1}(pd) + sqrt(rho) Phi^{-1}(q))/
sqrt(1-rho))``.  The mean loss equals ``pd``.

Reference: O. Vasicek (2002), "The distribution of loan portfolio value", Risk.
Fail-closed on parameters outside their valid ranges.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

Array = NDArray[np.float64]


def _check(pd: float, rho: float) -> None:
    if not 0.0 < pd < 1.0:
        raise ValueError("pd must be in (0, 1)")
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must be in (0, 1)")


def vasicek_loss_cdf(x: Array, pd: float, rho: float) -> Array:
    """CDF of the portfolio loss fraction."""
    _check(pd, rho)
    xx = np.clip(np.asarray(x, dtype=float), 1e-12, 1.0 - 1e-12)
    return np.asarray(
        norm.cdf((np.sqrt(1.0 - rho) * norm.ppf(xx) - norm.ppf(pd)) / np.sqrt(rho)), dtype=float
    )


def vasicek_loss_pdf(x: Array, pd: float, rho: float) -> Array:
    """Density of the portfolio loss fraction."""
    _check(pd, rho)
    xx = np.clip(np.asarray(x, dtype=float), 1e-12, 1.0 - 1e-12)
    ppf_x = norm.ppf(xx)
    inner = np.sqrt(1.0 - rho) * ppf_x - norm.ppf(pd)
    return np.asarray(
        np.sqrt((1.0 - rho) / rho) * np.exp(0.5 * ppf_x**2 - 0.5 / rho * inner**2), dtype=float
    )


def vasicek_var(q: float, pd: float, rho: float) -> float:
    """Loss quantile (Value-at-Risk of the loss fraction) at level ``q``."""
    _check(pd, rho)
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    return float(norm.cdf((norm.ppf(pd) + np.sqrt(rho) * norm.ppf(q)) / np.sqrt(1.0 - rho)))


def vasicek_es(q: float, pd: float, rho: float, n_nodes: int = 512) -> float:
    """Expected shortfall of the loss fraction beyond the ``q`` quantile."""
    _check(pd, rho)
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    us = np.linspace(q, 1.0 - 1e-9, n_nodes)
    losses = np.array([vasicek_var(float(u), pd, rho) for u in us])
    return float(np.trapezoid(losses, us) / (1.0 - q))
