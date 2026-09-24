"""Bachelier (1900) normal option-pricing model.

Under the Bachelier model the forward follows arithmetic Brownian motion, so
option prices are expressed with an *absolute* (normal) volatility ``sigma`` in
price units rather than a lognormal vol.  For forward ``F``, strike ``K``,
maturity ``T``, discount factor ``df`` and ``d = (F-K)/(sigma sqrt(T))``:

    call = df [ (F-K) Phi(d) + sigma sqrt(T) phi(d) ],
    put  = df [ (K-F) Phi(-d) + sigma sqrt(T) phi(d) ].

The at-the-money price is ``df sigma sqrt(T/(2 pi))``.  Implied normal vol is
recovered by bracketed root finding.

Reference: L. Bachelier (1900), "Theorie de la speculation."  Fail-closed on
non-positive vol/maturity or invalid option type.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def bachelier_price(
    forward: float,
    strike: float,
    maturity: float,
    sigma: float,
    option: str = "call",
    df: float = 1.0,
) -> float:
    """Bachelier (normal-model) option price."""
    if sigma <= 0.0 or maturity <= 0.0:
        raise ValueError("sigma and maturity must be positive")
    if option not in {"call", "put"}:
        raise ValueError("option must be 'call' or 'put'")
    vol = sigma * np.sqrt(maturity)
    d = (forward - strike) / vol
    if option == "call":
        val = (forward - strike) * norm.cdf(d) + vol * norm.pdf(d)
    else:
        val = (strike - forward) * norm.cdf(-d) + vol * norm.pdf(d)
    return float(df * val)


def bachelier_greeks(
    forward: float, strike: float, maturity: float, sigma: float, df: float = 1.0
) -> dict[str, float]:
    """Delta (call) and vega for the Bachelier model."""
    if sigma <= 0.0 or maturity <= 0.0:
        raise ValueError("sigma and maturity must be positive")
    vol = sigma * np.sqrt(maturity)
    d = (forward - strike) / vol
    return {
        "delta_call": float(df * norm.cdf(d)),
        "delta_put": float(df * (norm.cdf(d) - 1.0)),
        "vega": float(df * np.sqrt(maturity) * norm.pdf(d)),
    }


def bachelier_implied_vol(
    price: float,
    forward: float,
    strike: float,
    maturity: float,
    option: str = "call",
    df: float = 1.0,
) -> float:
    """Implied normal volatility from an option price."""
    if price <= 0.0 or maturity <= 0.0:
        raise ValueError("price and maturity must be positive")
    intrinsic = (
        df * max(forward - strike, 0.0) if option == "call" else df * max(strike - forward, 0.0)
    )
    if price < intrinsic - 1e-12:
        raise ValueError("price below intrinsic value")

    def obj(sig: float) -> float:
        return bachelier_price(forward, strike, maturity, sig, option, df) - price

    return float(brentq(obj, 1e-8, 1e6, xtol=1e-10))
