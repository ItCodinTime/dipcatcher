"""Barone-Adesi & Whaley (1987) American option approximation.

The BAW quadratic approximation writes the American price as the European price
plus an early-exercise premium proportional to a power of the underlying.  With
cost of carry ``b = r - q``, ``M = 2r/sigma^2``, ``N = 2b/sigma^2`` and
``Kc = 1 - e^{-rT}``:

    q2 = (-(N-1) + sqrt((N-1)^2 + 4 M/Kc)) / 2   (calls),
    q1 = (-(N-1) - sqrt((N-1)^2 + 4 M/Kc)) / 2   (puts),

with the critical exercise price found by root finding.  Below/above the
critical price the premium is added; beyond it the option is exercised.

Reference: G. Barone-Adesi, R. Whaley (1987), "Efficient analytic approximation
of American option values", Journal of Finance.  Fail-closed on invalid
parameters.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1(s: float, k: float, t: float, b: float, sigma: float) -> float:
    return (np.log(s / k) + (b + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))


def _bs(s: float, k: float, t: float, r: float, b: float, sigma: float, call: bool) -> float:
    d1 = _d1(s, k, t, b, sigma)
    d2 = d1 - sigma * np.sqrt(t)
    carry = np.exp((b - r) * t)
    if call:
        return float(s * carry * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2))
    return float(k * np.exp(-r * t) * norm.cdf(-d2) - s * carry * norm.cdf(-d1))


def _check(s: float, k: float, t: float, sigma: float) -> None:
    if s <= 0 or k <= 0 or t <= 0 or sigma <= 0:
        raise ValueError("require positive spot, strike, maturity and sigma")


def baw_american(
    s: float, k: float, t: float, r: float, q: float, sigma: float, option: str = "call"
) -> float:
    """Barone-Adesi-Whaley American option price."""
    _check(s, k, t, sigma)
    if option not in {"call", "put"}:
        raise ValueError("option must be 'call' or 'put'")
    b = r - q
    call = option == "call"
    euro = _bs(s, k, t, r, b, sigma, call)
    if call and b >= r:
        return euro  # never optimal to exercise a call early when b >= r
    m = 2.0 * r / sigma**2
    n = 2.0 * b / sigma**2
    kc = 1.0 - np.exp(-r * t)
    if kc <= 0.0:
        return euro
    disc = np.exp((b - r) * t)
    if call:
        q2 = (-(n - 1.0) + np.sqrt((n - 1.0) ** 2 + 4.0 * m / kc)) / 2.0

        def g(sc: float) -> float:
            prem = (1.0 - disc * norm.cdf(_d1(sc, k, t, b, sigma))) * sc / q2
            return sc - k - _bs(sc, k, t, r, b, sigma, True) - prem

        s_star = float(brentq(g, k * 1.0001, k * 50.0, xtol=1e-8))
        a2 = (s_star / q2) * (1.0 - disc * norm.cdf(_d1(s_star, k, t, b, sigma)))
        if s < s_star:
            return float(euro + a2 * (s / s_star) ** q2)
        return float(s - k)
    q1 = (-(n - 1.0) - np.sqrt((n - 1.0) ** 2 + 4.0 * m / kc)) / 2.0

    def h(sc: float) -> float:
        prem = (1.0 - disc * norm.cdf(-_d1(sc, k, t, b, sigma))) * sc / q1
        return k - sc - _bs(sc, k, t, r, b, sigma, False) + prem

    s_star = float(brentq(h, k * 1e-4, k * 0.9999, xtol=1e-8))
    a1 = -(s_star / q1) * (1.0 - disc * norm.cdf(-_d1(s_star, k, t, b, sigma)))
    if s > s_star:
        return float(euro + a1 * (s / s_star) ** q1)
    return float(k - s)
