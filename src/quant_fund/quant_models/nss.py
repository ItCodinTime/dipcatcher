"""Nelson–Siegel–Svensson yield curve.

davidalmeida90/quant-models ``yield-curve-us-brazil``. Instantaneous
forward and par-zero yields. Research curve engine, not a live bond book.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


def _factor(tau: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    x = tau / lam
    with np.errstate(divide="ignore", invalid="ignore"):
        level1 = np.where(np.abs(x) < 1e-12, 1.0, (1.0 - np.exp(-x)) / x)
    hump = level1 - np.exp(-x)
    return np.asarray(level1, dtype=float), np.asarray(hump, dtype=float)


def nss_yield(
    tau: ArrayLike,
    beta0: float,
    beta1: float,
    beta2: float,
    beta3: float,
    lam1: float,
    lam2: float,
) -> Array:
    """Continuously compounded zero yield for maturity ``tau`` (years)."""
    t = np.asarray(tau, dtype=float)
    if lam1 <= 0 or lam2 <= 0:
        raise ValueError("lambda1 and lambda2 must be positive")
    f1, h1 = _factor(t, lam1)
    _, h2 = _factor(t, lam2)
    y = beta0 + beta1 * f1 + beta2 * h1 + beta3 * h2
    return np.asarray(y, dtype=float)


def nss_forward(
    tau: ArrayLike,
    beta0: float,
    beta1: float,
    beta2: float,
    beta3: float,
    lam1: float,
    lam2: float,
) -> Array:
    """Instantaneous forward rate."""
    t = np.asarray(tau, dtype=float)
    if lam1 <= 0 or lam2 <= 0:
        raise ValueError("lambda1 and lambda2 must be positive")
    y1 = np.exp(-t / lam1)
    y2 = np.exp(-t / lam2)
    f = beta0 + beta1 * y1 + beta2 * (t / lam1) * y1 + beta3 * (t / lam2) * y2
    return np.asarray(f, dtype=float)


def nss_discount(tau: ArrayLike, *params: float) -> Array:
    t = np.asarray(tau, dtype=float)
    y = nss_yield(t, *params)
    return np.asarray(np.exp(-y * t), dtype=float)
