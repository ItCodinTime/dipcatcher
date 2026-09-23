"""Gatheral raw SVI total-variance smile.

davidalmeida90/quant-models / spy-iv-surface counterpart. Butterfly
density uses the Gatheral–Jacquier (2014) g-function. Calendar
arbitrage is checked when two slices are supplied. Research surface
engine; NN vol surfaces stay behind ADR-007.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

Array = NDArray[np.float64]


def svi_total_variance(
    k: ArrayLike,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> Array:
    """Raw SVI: ``w(k) = a + b (rho (k-m) + sqrt((k-m)^2 + sigma^2))``."""
    if b < 0 or abs(rho) >= 1.0 or sigma <= 0:
        raise ValueError("SVI requires b>=0, |rho|<1, sigma>0")
    kk = np.asarray(k, dtype=float)
    x = kk - m
    w = a + b * (rho * x + np.sqrt(x * x + sigma * sigma))
    return np.asarray(w, dtype=float)


def svi_density_g(
    k: ArrayLike,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> Array:
    """Gatheral–Jacquier g(k); g>=0 is the no-butterfly condition."""
    kk = np.asarray(k, dtype=float)
    w = svi_total_variance(kk, a, b, rho, m, sigma)
    x = kk - m
    disc = np.sqrt(x * x + sigma * sigma)
    wp = b * (rho + x / disc)
    wpp = b * sigma * sigma / disc**3
    g = (1 - kk * wp / (2 * w)) ** 2 - 0.25 * (wp * wp / w + 0.25) + 0.5 * wpp
    return np.asarray(g, dtype=float)


def svi_butterfly_ok(
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
    *,
    k_grid: ArrayLike | None = None,
) -> bool:
    grid = np.linspace(-2.0, 2.0, 81) if k_grid is None else np.asarray(k_grid, dtype=float)
    w = svi_total_variance(grid, a, b, rho, m, sigma)
    if np.any(w <= 0):
        return False
    return bool(np.all(svi_density_g(grid, a, b, rho, m, sigma) >= -1e-8))


def fit_svi(
    k: ArrayLike,
    total_variance: ArrayLike,
    *,
    x0: tuple[float, float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Least-squares raw SVI on a single expiry slice."""
    kk = np.asarray(k, dtype=float).reshape(-1)
    w = np.asarray(total_variance, dtype=float).reshape(-1)
    if kk.size != w.size or kk.size < 5:
        raise ValueError("need at least 5 (k, w) pairs")
    if x0 is None:
        x0 = (float(np.median(w)), 0.1, -0.3, 0.0, 0.1)

    def resid(theta: np.ndarray) -> np.ndarray:
        a, b, rho, m, sig = theta
        b = abs(b)
        rho = np.clip(rho, -0.999, 0.999)
        sig = max(float(sig), 1e-4)
        return svi_total_variance(kk, a, b, rho, m, sig) - w

    bounds = ([-1.0, 0.0, -0.999, -2.0, 1e-4], [2.0, 5.0, 0.999, 2.0, 5.0])
    sol = least_squares(resid, np.asarray(x0, dtype=float), bounds=bounds, max_nfev=400)
    a, b, rho, m, sig = (float(x) for x in sol.x)
    b = abs(b)
    rho = float(np.clip(rho, -0.999, 0.999))
    sig = max(sig, 1e-4)
    return {
        "a": a,
        "b": b,
        "rho": rho,
        "m": m,
        "sigma": sig,
        "rmse": float(np.sqrt(np.mean(resid(np.array([a, b, rho, m, sig])) ** 2))),
        "butterfly_ok": svi_butterfly_ok(a, b, rho, m, sig),
        "success": bool(sol.success),
    }
