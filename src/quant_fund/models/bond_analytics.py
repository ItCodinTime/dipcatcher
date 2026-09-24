"""Fixed-income analytics: price, yield, duration and convexity.

Uses continuous compounding: for cashflows ``c_i`` at times ``t_i`` and
continuously-compounded yield ``y``,

    P = sum_i c_i e^{-y t_i},
    Macaulay duration D = (1/P) sum_i t_i c_i e^{-y t_i},
    convexity C = (1/P) sum_i t_i^2 c_i e^{-y t_i}.

Under continuous compounding the modified duration equals the Macaulay duration
(``dP/dy = -D P``).  Yield-to-maturity is solved from the price by bracketed
root finding.

References: F. Macaulay (1938) (duration); F. Redington (1952) (convexity /
immunization).  Fail-closed on invalid cashflow specifications.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

Array = NDArray[np.float64]


def _check(times: Array, cashflows: Array) -> tuple[Array, Array]:
    t = np.asarray(times, dtype=float).ravel()
    c = np.asarray(cashflows, dtype=float).ravel()
    if t.size != c.size or t.size < 1 or not (np.isfinite(t).all() and np.isfinite(c).all()):
        raise ValueError("times and cashflows must be finite and aligned")
    if (t < 0).any():
        raise ValueError("times must be non-negative")
    return t, c


def bond_price(times: Array, cashflows: Array, ytm: float) -> float:
    """Present value of cashflows at continuously-compounded ``ytm``."""
    t, c = _check(times, cashflows)
    return float(np.sum(c * np.exp(-ytm * t)))


def yield_to_maturity(price: float, times: Array, cashflows: Array) -> float:
    """Continuously-compounded YTM that reprices the bond to ``price``."""
    if price <= 0.0:
        raise ValueError("price must be positive")
    t, c = _check(times, cashflows)
    if (c <= 0).all():
        raise ValueError("need at least one positive cashflow")

    def obj(y: float) -> float:
        return float(np.sum(c * np.exp(-y * t))) - price

    return float(brentq(obj, -0.5, 2.0, xtol=1e-10))


def macaulay_duration(times: Array, cashflows: Array, ytm: float) -> float:
    """Macaulay duration (continuous compounding)."""
    t, c = _check(times, cashflows)
    pv = c * np.exp(-ytm * t)
    total = float(pv.sum())
    if total <= 0.0:
        raise ValueError("non-positive present value")
    return float(np.sum(t * pv) / total)


def modified_duration(times: Array, cashflows: Array, ytm: float) -> float:
    """Modified duration; equals Macaulay duration under continuous compounding."""
    return macaulay_duration(times, cashflows, ytm)


def convexity(times: Array, cashflows: Array, ytm: float) -> float:
    """Convexity (second-order yield sensitivity, continuous compounding)."""
    t, c = _check(times, cashflows)
    pv = c * np.exp(-ytm * t)
    total = float(pv.sum())
    if total <= 0.0:
        raise ValueError("non-positive present value")
    return float(np.sum(t**2 * pv) / total)


def dv01(times: Array, cashflows: Array, ytm: float) -> float:
    """Dollar value of a 1bp yield move (price sensitivity)."""
    price = bond_price(times, cashflows, ytm)
    return float(modified_duration(times, cashflows, ytm) * price * 1e-4)
