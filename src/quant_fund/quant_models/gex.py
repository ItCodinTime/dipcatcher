"""Dealer gamma exposure and the last-half-hour GEX rule.

Math from davidalmeida90/gex-trading-bot (``gex.py``, ``last_hour.py``).
Sign convention is an assumption: dealers long calls, short puts.
No Cboe fetch and no IBKR / MES orders — decision only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quant_fund.quant_models.greeks import greeks

Array = NDArray[np.float64]

MULTIPLIER = 100.0
MES_MULT = 5.0


def bs_gamma(S: float, K: ArrayLike, T: ArrayLike, sigma: ArrayLike, r: float = 0.04) -> Array:
    """Black–Scholes gamma. ``T`` floored at four hours so expiry-day gamma is finite."""
    k = np.asarray(K, dtype=float)
    t = np.maximum(np.asarray(T, dtype=float), 4.0 / (365.0 * 24.0))
    sig = np.maximum(np.asarray(sigma, dtype=float), 1e-4)
    g = greeks(S, k, t, r, 0.0, sig, "call")
    return np.asarray(g["gamma"], dtype=float)


def gex_at(
    gamma: ArrayLike,
    open_interest: ArrayLike,
    side: ArrayLike,
    S: float,
    *,
    multiplier: float = MULTIPLIER,
) -> Array:
    """Dollar gamma per 1% move, dealer sign applied (calls +, puts −)."""
    g = np.asarray(gamma, dtype=float)
    oi = np.asarray(open_interest, dtype=float)
    sides = np.array([str(s).upper()[:1] for s in np.asarray(side).reshape(-1)])
    sign = np.where(sides == "C", 1.0, -1.0)
    return np.asarray(sign * g * oi * multiplier * S * S * 0.01, dtype=float)


def flip_level(
    strikes: ArrayLike,
    dte: ArrayLike,
    iv: ArrayLike,
    oi: ArrayLike,
    side: ArrayLike,
    spot: float,
    *,
    r: float = 0.04,
    lo: float = 0.85,
    hi: float = 1.15,
    n: int = 241,
) -> tuple[Array, Array, float | None]:
    """Total GEX on a spot grid and the zero-gamma crossing (linear interpolate)."""
    k = np.asarray(strikes, dtype=float)
    t = np.asarray(dte, dtype=float) / 365.0
    sig = np.asarray(iv, dtype=float)
    grid = np.linspace(spot * lo, spot * hi, n)
    total = np.array(
        [float(gex_at(bs_gamma(float(s), k, t, sig, r), oi, side, float(s)).sum()) for s in grid],
        dtype=float,
    )
    flip: float | None = None
    for i in range(len(grid) - 1):
        a, b = total[i], total[i + 1]
        if a == 0.0 or (a < 0 < b) or (a > 0 > b):
            flip = float(grid[i] + (grid[i + 1] - grid[i]) * (-a) / (b - a) if b != a else grid[i])
            break
    return np.asarray(grid, dtype=float), total, flip


@dataclass(frozen=True)
class LastHourDecision:
    action: str
    contracts: int
    why: str
    leg: str


def last_hour_decide(
    gex_prev: float,
    r_sofar: float,
    equity: float,
    spot: float,
    *,
    leverage: float = 1.0,
    fade_long_gamma: bool = True,
) -> LastHourDecision:
    """15:30 ET last-half-hour rule. No broker. Exit on the close, no overnight.

    Short gamma → go with the day (Baltussen et al. JFE 2021). Long gamma →
    fade the day (weaker leg; ``fade_long_gamma=False`` turns it off).
    """
    side = int(np.sign(r_sofar))
    n = int(equity * leverage // (MES_MULT * spot)) if spot > 0 else 0
    if side == 0 or n == 0:
        return LastHourDecision("FLAT", 0, "no move so far, or no size", "none")
    if gex_prev < 0:
        return LastHourDecision(
            "LONG" if side > 0 else "SHORT",
            n,
            "short gamma: dealers hedge with the move into the close, so go with the day",
            "follow",
        )
    if fade_long_gamma:
        return LastHourDecision(
            "SHORT" if side > 0 else "LONG",
            n,
            "long gamma: dealers lean against the move into the close, so go against the day",
            "fade",
        )
    return LastHourDecision("FLAT", 0, "long gamma and the fade leg is off", "none")
