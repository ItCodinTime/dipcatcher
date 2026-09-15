"""Almgren–Chriss discrete optimal trajectory."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def almgren_chriss_trajectory(
    quantity: float,
    n_slices: int,
    *,
    sigma: float,
    eta: float,
    gamma: float,
    risk_aversion: float,
    tau: float = 1.0,
) -> Array:
    """Holdings remaining after each slice, length n_slices+1, starts at quantity.

    Discrete AC with kappa^2 = lambda * sigma^2 / eta.
    As risk_aversion → 0, slices are nearly equal (TWAP-like).
    As risk_aversion increases, trajectory is more front-loaded.
    """
    x0 = float(quantity)
    if n_slices < 1:
        raise ValueError("n_slices >= 1")
    if x0 == 0:
        return np.zeros(n_slices + 1)
    t = np.arange(n_slices + 1, dtype=float) * tau
    t_end = t[-1]
    if risk_aversion <= 0 or eta <= 0 or sigma <= 0:
        # equal slices
        return x0 * (1.0 - t / t_end)
    kappa = np.sqrt(risk_aversion * sigma**2 / eta)
    # x(t) = x0 * sinh(kappa (T-t)) / sinh(kappa T)
    denom = np.sinh(kappa * t_end)
    if denom == 0 or not np.isfinite(denom):
        return x0 * (1.0 - t / t_end)
    x = x0 * np.sinh(kappa * (t_end - t)) / denom
    x[0] = x0
    x[-1] = 0.0
    return x


def slice_trades(holdings: Array) -> Array:
    return -np.diff(holdings)


def twap_trajectory(quantity: float, n_slices: int) -> Array:
    return almgren_chriss_trajectory(
        quantity, n_slices, sigma=1.0, eta=1.0, gamma=0.0, risk_aversion=0.0
    )


def front_loaded_trajectory(quantity: float, n_slices: int) -> Array:
    return almgren_chriss_trajectory(
        quantity, n_slices, sigma=0.02, eta=1e-6, gamma=0.0, risk_aversion=1e-2
    )


def expected_shortfall_ac(
    holdings: Array,
    trades: Array,
    *,
    arrival: float,
    eta: float,
    gamma: float,
    sigma: float,
    tau: float = 1.0,
) -> dict[str, float]:
    """Expected implementation shortfall under linear permanent + temp impact."""
    if abs(holdings[0]) < 1e-18:
        return {"expected_is": 0.0, "variance_is": 0.0, "expected_cost": 0.0}
    perm = gamma * np.sum(trades * (np.cumsum(trades) - trades / 2.0))
    temp = eta / tau * np.sum(trades**2)
    expected = perm + temp
    var = (sigma**2) * tau * np.sum(holdings[1:] ** 2)
    return {
        "expected_is": float(expected),
        "variance_is": float(var),
        "expected_cost": float(expected),
    }
