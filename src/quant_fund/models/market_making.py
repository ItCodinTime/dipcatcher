"""Avellaneda-Stoikov (2008) optimal market making.

Midprice dS = sigma dW, exponential utility with risk aversion gamma,
market-order arrival intensity lambda(delta) = A exp(-kappa delta).
Closed-form asymptotic quotes:

  reservation price  r = s - q gamma sigma^2 (T - t)
  optimal half-spread delta = (gamma sigma^2 (T-t) + (2/gamma) ln(1 + gamma/kappa)) / 2
  bid = r - delta, ask = r + delta.

Also: arrival-intensity calibration from (depth, fill-rate) pairs and
an inventory-risk decomposition. Fail-closed on invalid params.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _check_mm(gamma: float, sigma: float, tau: float, kappa: float | None = None) -> None:
    vals = [gamma, sigma, tau] + ([] if kappa is None else [kappa])
    if not np.isfinite(vals).all():
        raise ValueError("parameters must be finite")
    if gamma <= 0.0 or sigma <= 0.0 or tau < 0.0:
        raise ValueError("gamma, sigma > 0, tau >= 0")
    if kappa is not None and kappa <= 0.0:
        raise ValueError("kappa must be > 0")


def as_reservation_price(
    mid: Array | float,
    inventory: Array | float,
    gamma: float,
    sigma: float,
    tau: float,
) -> Array:
    """r(s, q, t) = s - q gamma sigma^2 (T - t). Vectorized in q/mid."""
    _check_mm(gamma, sigma, tau)
    s = np.asarray(mid, dtype=float)
    q = np.asarray(inventory, dtype=float)
    if not np.isfinite(s).all() or not np.isfinite(q).all():
        raise ValueError("mid and inventory must be finite")
    return np.asarray(s - q * gamma * sigma * sigma * tau, dtype=float)


def as_optimal_spread(gamma: float, sigma: float, tau: float, kappa: float) -> float:
    """Total optimal spread delta_a + delta_b (AS closed-form approx)."""
    _check_mm(gamma, sigma, tau, kappa)
    return float(gamma * sigma * sigma * tau + (2.0 / gamma) * math.log(1.0 + gamma / kappa))


def as_optimal_quotes(
    mid: float,
    inventory: float,
    gamma: float,
    sigma: float,
    tau: float,
    kappa: float,
) -> dict[str, float]:
    """Optimal bid/ask around the reservation price."""
    _check_mm(gamma, sigma, tau, kappa)
    if not np.isfinite([mid, inventory]).all():
        raise ValueError("mid and inventory must be finite")
    r = float(as_reservation_price(mid, inventory, gamma, sigma, tau))
    half = as_optimal_spread(gamma, sigma, tau, kappa) / 2.0
    return {
        "reservation_price": r,
        "bid": r - half,
        "ask": r + half,
        "half_spread": half,
        "skew": r - float(mid),
    }


def estimate_arrival_intensity(depths: Array, fill_rates: Array) -> dict[str, float]:
    """Calibrate lambda(delta) = A exp(-kappa delta) by regressing
    ln(fill_rate) on quote depth. ``fill_rates`` = fills per unit time
    at each quoted depth; zeros are dropped (log undefined)."""
    d = np.asarray(depths, dtype=float).ravel()
    lam = np.asarray(fill_rates, dtype=float).ravel()
    if d.shape != lam.shape or d.size < 3:
        raise ValueError("depths and fill_rates must match, >= 3 points")
    if not np.isfinite(d).all() or not np.isfinite(lam).all():
        raise ValueError("inputs must be finite")
    pos = lam > 0.0
    if pos.sum() < 3:
        raise ValueError("need >= 3 positive fill rates")
    x = d[pos]
    y = np.log(lam[pos])
    a, b = np.polyfit(x, y, 1)  # ln lam = ln A - kappa delta
    if a >= 0.0:
        raise ValueError("fill rates must decay with depth (slope < 0)")
    return {"A": float(math.exp(b)), "kappa": float(-a), "n": float(pos.sum())}


def as_inventory_bounds(
    q_max: float,
    gamma: float,
    sigma: float,
    tau: float,
    kappa: float,
) -> dict[str, float]:
    """Inventory-scaled quote envelope: quotes at q = -q_max vs +q_max.

    Gives the extreme reservation-price range the market maker will
    quote over the inventory band — a practical inventory-risk measure.
    """
    _check_mm(gamma, sigma, tau, kappa)
    if not np.isfinite(q_max) or q_max <= 0.0:
        raise ValueError("q_max must be > 0")
    r_lo = float(as_reservation_price(0.0, q_max, gamma, sigma, tau))
    r_hi = float(as_reservation_price(0.0, -q_max, gamma, sigma, tau))
    return {
        "reservation_range": r_hi - r_lo,
        "skew_per_unit": gamma * sigma * sigma * tau,
        "spread": as_optimal_spread(gamma, sigma, tau, kappa),
    }
