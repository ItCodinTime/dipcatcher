"""Johnson SB (bounded) and SL (lognormal) translation distributions.

Johnson (1949) proposed translating a variable to a standard normal ``z``:

- **SB** (bounded on ``(xi, xi+lambda)``): ``z = gamma + delta log(u/(1-u))``
  with ``u = (x-xi)/lambda``.
- **SL** (lognormal, semi-bounded on ``x > xi``): ``z = gamma + delta log(x-xi)``.

Given the support (estimated from the data range for SB, or ``min`` for SL) the
shape parameters ``gamma`` and ``delta`` follow from a normal-scores regression
of ``Phi^{-1}(rank/(n+1))`` on the transformed variable, a simple special case
of the Hill-Hill-Holder (1976) fitting approach.

References: N. L. Johnson (1949), Biometrika; I. D. Hill, R. Hill, R. L. Holder
(1976), Applied Statistics.  Fail-closed on invalid parameters or non-finite
input.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

Array = NDArray[np.float64]


# ---------------------------------------------------------------------------- SB
def johnson_sb_pdf(x: Array, gamma: float, delta: float, xi: float, lam: float) -> Array:
    """Johnson SB density on (xi, xi+lam)."""
    if delta <= 0.0 or lam <= 0.0:
        raise ValueError("delta and lambda must be positive")
    xa = np.asarray(x, dtype=float)
    u = (xa - xi) / lam
    inside = (u > 0.0) & (u < 1.0)
    out = np.zeros_like(xa)
    uu = u[inside]
    z = gamma + delta * np.log(uu / (1.0 - uu))
    out[inside] = delta / (lam * np.sqrt(2.0 * np.pi)) / (uu * (1.0 - uu)) * np.exp(-0.5 * z**2)
    return out


def johnson_sb_cdf(x: Array, gamma: float, delta: float, xi: float, lam: float) -> Array:
    """Johnson SB CDF."""
    if delta <= 0.0 or lam <= 0.0:
        raise ValueError("delta and lambda must be positive")
    u = np.clip((np.asarray(x, dtype=float) - xi) / lam, 1e-12, 1.0 - 1e-12)
    return norm.cdf(gamma + delta * np.log(u / (1.0 - u)))


def johnson_sb_ppf(p: float, gamma: float, delta: float, xi: float, lam: float) -> float:
    """Johnson SB quantile."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    z = norm.ppf(p)
    u = 1.0 / (1.0 + np.exp(-(z - gamma) / delta))
    return float(xi + lam * u)


def johnson_sb_fit(x: Array, margin: float = 0.01) -> dict[str, float]:
    """Fit SB by fixing the support to the data range and regressing normal scores."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < 20 or not np.isfinite(arr).all():
        raise ValueError("x must be finite with >= 20 observations")
    span = float(arr.max() - arr.min())
    if span <= 0.0:
        raise ValueError("data has zero range")
    xi = float(arr.min() - margin * span)
    lam = float(span * (1.0 + 2.0 * margin))
    u = (np.sort(arr) - xi) / lam
    ranks = (np.arange(1, arr.size + 1) - 0.5) / arr.size
    z = norm.ppf(ranks)
    feat = np.log(u / (1.0 - u))
    delta, gamma = np.polyfit(feat, z, 1)
    return {"gamma": float(gamma), "delta": float(delta), "xi": xi, "lam": lam}


# ---------------------------------------------------------------------------- SL
def johnson_sl_pdf(x: Array, gamma: float, delta: float, xi: float) -> Array:
    """Johnson SL (three-parameter lognormal) density on x > xi."""
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    xa = np.asarray(x, dtype=float)
    out = np.zeros_like(xa)
    mask = xa > xi
    d = xa[mask] - xi
    z = gamma + delta * np.log(d)
    out[mask] = delta / (d * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z**2)
    return out


def johnson_sl_cdf(x: Array, gamma: float, delta: float, xi: float) -> Array:
    """Johnson SL CDF."""
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    xa = np.asarray(x, dtype=float)
    d = np.clip(xa - xi, 1e-12, None)
    return np.where(xa > xi, norm.cdf(gamma + delta * np.log(d)), 0.0)


def johnson_sl_ppf(p: float, gamma: float, delta: float, xi: float) -> float:
    """Johnson SL quantile."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    return float(xi + np.exp((norm.ppf(p) - gamma) / delta))


def johnson_sl_fit(x: Array, margin: float = 0.01) -> dict[str, float]:
    """Fit SL by fixing ``xi`` below the minimum and regressing normal scores."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < 20 or not np.isfinite(arr).all():
        raise ValueError("x must be finite with >= 20 observations")
    span = float(arr.max() - arr.min())
    xi = float(arr.min() - margin * (span if span > 0 else 1.0))
    d = np.sort(arr) - xi
    ranks = (np.arange(1, arr.size + 1) - 0.5) / arr.size
    z = norm.ppf(ranks)
    delta, gamma = np.polyfit(np.log(d), z, 1)
    return {"gamma": float(gamma), "delta": float(delta), "xi": xi}
