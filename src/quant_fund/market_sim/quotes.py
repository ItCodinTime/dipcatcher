"""Avellaneda–Stoikov quotes in tick space.

The horizon ``T - t`` is normalized to one unit, and ``sigma2`` is the
variance of one-step mid changes in ticks. Inventory is in units of the
quote size, not raw shares. This is a quoting rule for the simulator, not
a forecast and not a live order.
"""

from __future__ import annotations

import math


def avellaneda_stoikov_quotes(
    mid_tick: float,
    inventory_units: float,
    gamma: float,
    k: float,
    sigma2: float,
) -> tuple[int, int, float]:
    """Return ``(bid_tick, ask_tick, half_spread_ticks)``.

    Avellaneda and Stoikov (2008), reservation price and optimal spread:

        r = s - q γ σ²
        spread = γ σ² + (2 / γ) log(1 + γ / k)

    The quoted half-spread is half of that, and at least one tick.
    """
    if not math.isfinite(mid_tick):
        raise ValueError("mid_tick must be finite")
    if not math.isfinite(inventory_units):
        raise ValueError("inventory_units must be finite")
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be finite and positive")
    if not math.isfinite(k) or k <= 0.0:
        raise ValueError("k must be finite and positive")
    if not math.isfinite(sigma2) or sigma2 < 0.0:
        raise ValueError("sigma2 must be finite and non-negative")
    variance = max(float(sigma2), 1e-8)
    reservation = float(mid_tick) - float(inventory_units) * gamma * variance
    full_spread = gamma * variance + (2.0 / gamma) * math.log(1.0 + gamma / k)
    half = max(0.5 * full_spread, 1.0)
    bid = math.floor(reservation - half)
    ask = math.ceil(reservation + half)
    if ask <= bid:
        ask = bid + 1
    return int(bid), int(ask), float(half)
