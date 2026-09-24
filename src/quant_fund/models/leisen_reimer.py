"""Leisen-Reimer (1996) binomial option tree.

The Leisen-Reimer tree chooses up/down moves and the risk-neutral probability by
inverting the Black-Scholes ``d1``/``d2`` through a Peizer-Pratt normal
approximation, so the tree centres on the strike and converges much faster
(order 1/n^2, monotonically) than the Cox-Ross-Rubinstein tree.  The step count
``n`` must be odd.

Reference: D. Leisen, M. Reimer (1996), "Binomial models for option valuation --
examining and improving convergence", Applied Mathematical Finance.  Fail-closed
on invalid parameters.
"""

from __future__ import annotations

import numpy as np


def _peizer_pratt(z: float, n: int) -> float:
    """Peizer-Pratt method-2 inversion of the normal CDF onto (0, 1)."""
    c = z / (n + 1.0 / 3.0 + 0.1 / (n + 1.0))
    return 0.5 + np.sign(z) * 0.5 * np.sqrt(1.0 - np.exp(-(c**2) * (n + 1.0 / 6.0)))


def leisen_reimer(
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    n: int = 101,
    option: str = "call",
    american: bool = False,
) -> float:
    """Leisen-Reimer binomial price (European or American)."""
    if s <= 0 or k <= 0 or t <= 0 or sigma <= 0:
        raise ValueError("require positive spot, strike, maturity and sigma")
    if option not in {"call", "put"}:
        raise ValueError("option must be 'call' or 'put'")
    if n < 3:
        raise ValueError("n must be >= 3")
    if n % 2 == 0:
        n += 1  # Leisen-Reimer requires an odd number of steps
    dt = t / n
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    p = _peizer_pratt(d2, n)
    p_prime = _peizer_pratt(d1, n)
    growth = np.exp((r - q) * dt)
    u = growth * p_prime / p
    d = (growth - p * u) / (1.0 - p)
    disc = np.exp(-r * dt)
    j = np.arange(n + 1, dtype=float)
    spot = s * u ** (n - j) * d**j
    call = option == "call"
    value = np.maximum(spot - k, 0.0) if call else np.maximum(k - spot, 0.0)
    for i in range(n - 1, -1, -1):
        value = disc * (p * value[: i + 1] + (1.0 - p) * value[1 : i + 2])
        if american:
            j_i = np.arange(i + 1, dtype=float)
            spot_i = s * u ** (i - j_i) * d**j_i
            intrinsic = np.maximum(spot_i - k, 0.0) if call else np.maximum(k - spot_i, 0.0)
            value = np.maximum(value, intrinsic)
    return float(value[0])
