"""Vasicek and Cox-Ingersoll-Ross one-factor short-rate models.

Both are affine term-structure models with closed-form zero-coupon bond prices
``P(tau) = A(tau) exp(-B(tau) r)``:

- **Vasicek** (1977): ``dr = kappa(theta - r) dt + sigma dW``; Gaussian, allows
  negative rates, exact discretisation as an AR(1).
- **CIR** (Cox, Ingersoll & Ross 1985): ``dr = kappa(theta - r) dt +
  sigma sqrt(r) dW``; non-negative under the Feller condition
  ``2 kappa theta >= sigma^2``.

Calibration uses the exact AR(1) discretisation (Vasicek) and its
heteroscedastic analogue (CIR) via ordinary least squares on the drift.
Fail-closed on invalid parameters or non-finite input.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def vasicek_bond_price(r0: float, tau: float, kappa: float, theta: float, sigma: float) -> float:
    """Vasicek zero-coupon bond price P(tau)."""
    if kappa <= 0.0 or sigma <= 0.0 or tau < 0.0:
        raise ValueError("require kappa > 0, sigma > 0, tau >= 0")
    b = (1.0 - np.exp(-kappa * tau)) / kappa
    a = np.exp((theta - sigma**2 / (2.0 * kappa**2)) * (b - tau) - sigma**2 * b**2 / (4.0 * kappa))
    return float(a * np.exp(-b * r0))


def vasicek_simulate(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    n: int,
    dt: float = 1.0 / 252.0,
    rng: np.random.Generator | None = None,
) -> Array:
    """Exact-discretisation Vasicek short-rate path."""
    if kappa <= 0.0 or sigma <= 0.0:
        raise ValueError("require kappa > 0 and sigma > 0")
    gen = np.random.default_rng() if rng is None else rng
    r = np.empty(n)
    r[0] = r0
    e = np.exp(-kappa * dt)
    sd = sigma * np.sqrt((1.0 - e**2) / (2.0 * kappa))
    z = gen.standard_normal(n)
    for t in range(1, n):
        r[t] = r[t - 1] * e + theta * (1.0 - e) + sd * z[t]
    return r


def vasicek_calibrate(r: Array, dt: float = 1.0 / 252.0) -> dict[str, float]:
    """OLS calibration of Vasicek from a short-rate series."""
    arr = np.asarray(r, dtype=float).ravel()
    if arr.size < 30 or not np.isfinite(arr).all():
        raise ValueError("series must be finite with >= 30 observations")
    x, y = arr[:-1], arr[1:]
    b, a = np.polyfit(x, y, 1)
    if not 0.0 < b < 1.0:
        raise ValueError("degenerate AR(1) slope; cannot calibrate")
    kappa = -np.log(b) / dt
    theta = a / (1.0 - b)
    resid = y - (a + b * x)
    var_e = float(resid.var(ddof=2))
    sigma = float(np.sqrt(var_e * 2.0 * kappa / (1.0 - b**2)))
    return {"kappa": float(kappa), "theta": float(theta), "sigma": sigma}


def cir_bond_price(r0: float, tau: float, kappa: float, theta: float, sigma: float) -> float:
    """CIR zero-coupon bond price P(tau)."""
    if kappa <= 0.0 or sigma <= 0.0 or theta <= 0.0 or tau < 0.0:
        raise ValueError("require kappa, theta, sigma > 0 and tau >= 0")
    g = np.sqrt(kappa**2 + 2.0 * sigma**2)
    denom = (g + kappa) * (np.exp(g * tau) - 1.0) + 2.0 * g
    b = 2.0 * (np.exp(g * tau) - 1.0) / denom
    a = (2.0 * g * np.exp((kappa + g) * tau / 2.0) / denom) ** (2.0 * kappa * theta / sigma**2)
    return float(a * np.exp(-b * r0))


def cir_simulate(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    n: int,
    dt: float = 1.0 / 252.0,
    rng: np.random.Generator | None = None,
) -> Array:
    """Full-truncation Euler CIR short-rate path (non-negative)."""
    if kappa <= 0.0 or sigma <= 0.0 or theta <= 0.0:
        raise ValueError("require kappa, theta, sigma > 0")
    gen = np.random.default_rng() if rng is None else rng
    r = np.empty(n)
    r[0] = r0
    z = gen.standard_normal(n)
    for t in range(1, n):
        rp = max(r[t - 1], 0.0)
        r[t] = rp + kappa * (theta - rp) * dt + sigma * np.sqrt(rp * dt) * z[t]
    return np.maximum(r, 0.0)


def cir_calibrate(r: Array, dt: float = 1.0 / 252.0) -> dict[str, float]:
    """OLS calibration of CIR using the heteroscedastic drift regression."""
    arr = np.asarray(r, dtype=float).ravel()
    if arr.size < 30 or not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError("series must be finite, non-negative, >= 30 observations")
    x, y = arr[:-1], arr[1:]
    b, a = np.polyfit(x, y, 1)
    if not 0.0 < b < 1.0:
        raise ValueError("degenerate slope; cannot calibrate")
    kappa = (1.0 - b) / dt
    theta = a / (1.0 - b)
    resid = y - (a + b * x)
    # Var(resid) ~ sigma^2 r dt; regress squared resid on x through origin.
    denom = float(np.dot(x, x))
    sigma2 = float(np.dot(resid**2, x) / denom / dt) if denom > 0 else float("nan")
    return {"kappa": float(kappa), "theta": float(theta), "sigma": float(np.sqrt(max(sigma2, 0.0)))}
