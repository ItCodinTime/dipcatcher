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
    if not np.isfinite(x0):
        raise ValueError("quantity must be finite")
    if isinstance(n_slices, bool) or int(n_slices) != n_slices or int(n_slices) < 1:
        raise ValueError("n_slices must be an integer >= 1")
    n_slices = int(n_slices)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and > 0")
    if not all(np.isfinite(v) for v in (sigma, eta, gamma, risk_aversion)):
        raise ValueError("sigma, eta, gamma, risk_aversion must be finite")
    if eta <= 0 or sigma <= 0:
        raise ValueError("eta and sigma must be > 0")
    if risk_aversion < 0:
        raise ValueError("risk_aversion must be >= 0")
    if x0 == 0:
        return np.zeros(n_slices + 1)
    t: Array = np.arange(n_slices + 1, dtype=np.float64) * tau
    t_end = float(t[-1])
    if risk_aversion == 0:
        # AC limit at zero risk aversion is exactly TWAP.
        return x0 * (1.0 - t / t_end)
    kappa = float(np.sqrt(risk_aversion * sigma**2 / eta))
    # x(t) = x0 * sinh(kappa (T-t)) / sinh(kappa T), evaluated in the stable
    # exponential form e^{-kt}(1 - e^{-2k(T-t)})/(1 - e^{-2kT}) so large kappa*T
    # yields the true AC limit (immediate liquidation) instead of overflowing.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        x = (
            x0
            * np.exp(-kappa * t)
            * (-np.expm1(-2.0 * kappa * (t_end - t)))
            / (-np.expm1(-2.0 * kappa * t_end))
        )
    x[0] = x0
    x[-1] = 0.0
    if not np.all(np.isfinite(x)):
        raise ValueError("almgren_chriss trajectory produced non-finite holdings")
    return np.asarray(x, dtype=np.float64)


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
    h = np.asarray(holdings, dtype=float).reshape(-1)
    tr = np.asarray(trades, dtype=float).reshape(-1)
    if h.size < 2:
        raise ValueError("holdings length must be >= 2")
    if tr.size != h.size - 1:
        raise ValueError("trades length must equal len(holdings)-1")
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and > 0")
    if not all(np.isfinite(v) for v in (eta, gamma, sigma, arrival)):
        raise ValueError("eta, gamma, sigma, arrival must be finite")
    # No zero-parent shortcut: a flat book with no trades already yields zero
    # from the formulas, while a flat start with non-zero trades is incoherent
    # and must not be silently reported as zero cost.
    perm = gamma * np.sum(tr * (np.cumsum(tr) - tr / 2.0))
    temp = eta / tau * np.sum(tr**2)
    expected = perm + temp
    var = (sigma**2) * tau * np.sum(h[1:] ** 2)
    return {
        "expected_is": float(expected),
        "variance_is": float(var),
        "expected_cost": float(expected),
    }
