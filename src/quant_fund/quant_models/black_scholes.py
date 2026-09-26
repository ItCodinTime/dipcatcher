"""Black–Scholes–Merton price, no-arbitrage bounds, implied vol.

Faithful to davidalmeida90/quant-models ``black-scholes/model.py`` with
continuous dividend yield ``q``. Research/pricing engine only.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.stats import norm

Array = NDArray[np.float64]


def d1(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike, sigma: ArrayLike
) -> Array:
    S, K, T, r, q, sigma = (np.asarray(z, dtype=float) for z in (S, K, T, r, q, sigma))
    vol_sqrt = sigma * np.sqrt(T)
    return np.asarray((np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt, dtype=float)


def d2(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike, sigma: ArrayLike
) -> Array:
    return np.asarray(
        d1(S, K, T, r, q, sigma)
        - np.asarray(sigma, dtype=float) * np.sqrt(np.asarray(T, dtype=float)),
        dtype=float,
    )


def bs_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
    option_type: str = "call",
) -> Array:
    """BSM price. ``option_type`` is ``call`` or ``put``."""
    kind = str(option_type).lower()
    S, K, T, r, q, sigma = (np.asarray(z, dtype=float) for z in (S, K, T, r, q, sigma))
    d1_val = d1(S, K, T, r, q, sigma)
    d2_val = d2(S, K, T, r, q, sigma)
    df_q = np.exp(-q * T)
    df_r = np.exp(-r * T)
    if kind == "call":
        price = S * df_q * norm.cdf(d1_val) - K * df_r * norm.cdf(d2_val)
    elif kind == "put":
        price = K * df_r * norm.cdf(-d2_val) - S * df_q * norm.cdf(-d1_val)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    return np.asarray(price, dtype=float)


def price_bounds(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    option_type: str = "call",
) -> tuple[Array, Array]:
    """European no-arbitrage bounds: call in ``[max(F_S - F_K, 0), F_S]``."""
    kind = str(option_type).lower()
    S, K, T, r, q = (np.asarray(z, dtype=float) for z in (S, K, T, r, q))
    pv_s = S * np.exp(-q * T)
    pv_k = K * np.exp(-r * T)
    if kind == "call":
        lower = np.maximum(pv_s - pv_k, 0.0)
        upper = pv_s
    elif kind == "put":
        lower = np.maximum(pv_k - pv_s, 0.0)
        upper = pv_k
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def implied_volatility(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    market_price: float,
    option_type: str = "call",
    *,
    sigma_lo: float = 1e-6,
    sigma_hi: float = 10.0,
    tol: float = 1e-12,
) -> float:
    """Brent implied vol. Raises if the quote is outside no-arbitrage bounds."""
    lower, upper = price_bounds(S, K, T, r, q, option_type)
    lo = float(np.asarray(lower).reshape(-1)[0])
    hi = float(np.asarray(upper).reshape(-1)[0])
    if market_price < lo:
        raise ValueError(f"Market price {market_price:.6g} < lower bound {lo:.6g}")
    if market_price > hi:
        raise ValueError(f"Market price {market_price:.6g} > upper bound {hi:.6g}")

    def objective(sigma: float) -> float:
        return float(bs_price(S, K, T, r, q, sigma, option_type)) - market_price

    return float(brentq(objective, sigma_lo, sigma_hi, xtol=tol))


def put_call_parity_gap(
    call: ArrayLike,
    put: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
) -> Array:
    """``C - P - (S e^{-qT} - K e^{-rT})``. Zero (up to fp) under BSM."""
    call, put, S, K, T, r, q = (np.asarray(z, dtype=float) for z in (call, put, S, K, T, r, q))
    return np.asarray(call - put - (S * np.exp(-q * T) - K * np.exp(-r * T)), dtype=float)
