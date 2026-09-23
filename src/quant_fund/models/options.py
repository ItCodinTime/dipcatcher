"""Black–Scholes–Merton option pricing, Greeks, and implied volatility.

References:
- Black & Scholes (1973); Merton (1973): the BSM formula.
- Manaster & Koehler (1982): Newton implied-vol seed.
- Brenner & Subrahmanyam (1988): ATM implied-vol approximation.
- Breeden & Litzenberger (1978): risk-neutral density from call prices.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Array = NDArray[np.float64]


def _check(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> None:
    if not (np.isfinite(S) and S > 0 and np.isfinite(K) and K > 0):
        raise ValueError("S and K must be positive and finite")
    if not (np.isfinite(T) and T > 0):
        raise ValueError("T must be positive")
    if not (np.isfinite(sigma) and sigma > 0):
        raise ValueError("sigma must be positive")
    if not np.isfinite(r):
        raise ValueError("r must be finite")


def bs_price(
    S: float, K: float, T: float, sigma: float, r: float = 0.0, call: bool = True
) -> float:
    """BSM European option price (Black–Scholes–Merton)."""
    _check(S, K, T, sigma, r)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if call:
        return float(S * stats.norm.cdf(d1) - K * math.exp(-r * T) * stats.norm.cdf(d2))
    return float(K * math.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1))


def bs_greeks(
    S: float, K: float, T: float, sigma: float, r: float = 0.0, call: bool = True
) -> dict[str, float]:
    """Delta, gamma, vega (per 1 vol point... per unit sigma), theta/yr, rho."""
    _check(S, K, T, sigma, r)
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / sq
    d2 = d1 - sq
    pdf = float(stats.norm.pdf(d1))
    df = math.exp(-r * T)
    if call:
        delta = float(stats.norm.cdf(d1))
        theta = -S * pdf * sigma / (2 * math.sqrt(T)) - r * K * df * stats.norm.cdf(d2)
        rho = K * T * df * float(stats.norm.cdf(d2))
    else:
        delta = float(stats.norm.cdf(d1) - 1.0)
        theta = -S * pdf * sigma / (2 * math.sqrt(T)) + r * K * df * stats.norm.cdf(-d2)
        rho = -K * T * df * float(stats.norm.cdf(-d2))
    return {
        "delta": delta,
        "gamma": float(pdf / (S * sq)),
        "vega": float(S * pdf * math.sqrt(T)),
        "theta": float(theta),
        "rho": float(rho),
    }


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.0,
    call: bool = True,
    tol: float = 1e-8,
) -> float:
    """Implied volatility via Brent bracketing on [1e-6, 5].

    Fail-closed on prices outside no-arbitrage bounds."""
    if not (np.isfinite(price) and price >= 0):
        raise ValueError("price must be nonnegative and finite")
    df = math.exp(-r * T)
    if call:
        lo_b = max(S - K * df, 0.0)
        hi_b = S
    else:
        lo_b = max(K * df - S, 0.0)
        hi_b = K * df
    if price < lo_b - 1e-10 or price > hi_b + 1e-10:
        raise ValueError("price outside no-arbitrage bounds")

    def f(sig: float) -> float:
        return bs_price(S, K, T, sig, r, call) - price

    from scipy import optimize as opt

    try:
        return float(opt.brentq(f, 1e-6, 5.0, xtol=tol, rtol=1e-10))
    except ValueError as exc:
        raise ValueError("implied vol bracketing failed") from exc


def put_call_parity_gap(
    call_price: float, put_price: float, S: float, K: float, T: float, r: float = 0.0
) -> float:
    """Breeden-style parity check: C - P - (S - K e^{-rT})."""
    for v in (call_price, put_price, S, K, T, r):
        if not np.isfinite(v):
            raise ValueError("inputs must be finite")
    return float(call_price - put_price - (S - K * math.exp(-r * T)))


def risk_neutral_density(
    strikes: Array, call_prices: Array, T: float, r: float = 0.0
) -> dict[str, Array]:
    """Breeden–Litzenberger (1978): RN density = e^{rT} d^2C/dK^2,
    estimated by second differences on an ordered strike grid."""
    K = np.asarray(strikes, dtype=float).reshape(-1)
    C = np.asarray(call_prices, dtype=float).reshape(-1)
    if K.size != C.size or K.size < 5:
        raise ValueError("need >= 5 aligned strikes/prices")
    if not (np.all(np.isfinite(K)) and np.all(np.isfinite(C))):
        raise ValueError("inputs must be finite")
    order = np.argsort(K)
    K = K[order]
    C = C[order]
    if np.any(np.diff(K) <= 0):
        raise ValueError("strikes must be strictly increasing")
    d2 = np.empty(K.size - 2)
    for i in range(1, K.size - 1):
        h1 = K[i] - K[i - 1]
        h2 = K[i + 1] - K[i]
        d2[i - 1] = 2.0 * ((C[i + 1] - C[i]) / h2 - (C[i] - C[i - 1]) / h1) / (h1 + h2)
    dens = np.maximum(d2 * math.exp(r * T), 0.0)
    return {"strikes": K[1:-1], "density": dens}
