"""Closed-form implied-volatility approximations.

- **Brenner-Subrahmanyam** (1988): for an at-the-money option the price is nearly
  linear in volatility, giving ``sigma ~ sqrt(2 pi / T) * C / S``.
- **Corrado-Miller** (1996): a quadratic correction that stays accurate away from
  the money,

    sigma = sqrt(2 pi/T)/(S + K e^{-rT}) *
            [ (C - X/2) + sqrt( (C - X/2)^2 - X^2/pi ) ],   X = S - K e^{-rT}.

These avoid the iterative root find used by the exact implied vol while
remaining close to it near the money.

References: M. Brenner, M. Subrahmanyam (1988), Financial Analysts Journal;
C. Corrado, T. Miller (1996), Journal of Banking & Finance.  Fail-closed on
non-positive inputs.
"""

from __future__ import annotations

import numpy as np


def brenner_subrahmanyam_iv(call_price: float, s: float, t: float) -> float:
    """At-the-money implied volatility approximation (forward-ATM)."""
    if call_price <= 0.0 or s <= 0.0 or t <= 0.0:
        raise ValueError("call_price, s and t must be positive")
    return float(np.sqrt(2.0 * np.pi / t) * call_price / s)


def corrado_miller_iv(call_price: float, s: float, k: float, t: float, r: float = 0.0) -> float:
    """Corrado-Miller (1996) closed-form implied volatility from a call price."""
    if call_price <= 0.0 or s <= 0.0 or k <= 0.0 or t <= 0.0:
        raise ValueError("prices, strike and maturity must be positive")
    x = s - k * np.exp(-r * t)
    diff = call_price - x / 2.0
    radicand = diff**2 - x**2 / np.pi
    radicand = max(radicand, 0.0)  # guard the deep-ITM/OTM regime
    return float(np.sqrt(2.0 * np.pi / t) / (s + k * np.exp(-r * t)) * (diff + np.sqrt(radicand)))
