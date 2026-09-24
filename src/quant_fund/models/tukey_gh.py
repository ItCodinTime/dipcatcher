"""Tukey (1977) g-and-h distribution with Hoaglin (1985) quantile fitting.

The g-and-h family is defined through a monotone transform of a standard
normal ``Z``:

    x = A + B * (exp(g Z) - 1) / g * exp(h Z^2 / 2),   h >= 0,

with the ``g -> 0`` limit ``x = A + B Z exp(h Z^2 / 2)``.  ``g`` controls
skewness and ``h`` controls tail weight; ``g = h = 0`` is the normal.  Because
the transform is monotone in ``Z`` the quantile function is closed form and the
density follows from the change of variables ``f(x) = phi(Z) / (dx/dZ)``.

Fitting uses Hoaglin's quantile ("letter value") method: ``g`` from the
log-ratio of upper/lower half spreads and ``(ln B, h)`` from a regression of the
corrected full spread on ``Z^2/2``.

References: J. W. Tukey (1977); D. C. Hoaglin (1985), in *Exploring Data Tables,
Trends, and Shapes*.  Fail-closed on non-finite input or non-positive scale.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.stats import norm

Array = NDArray[np.float64]


def _transform(z: Array, a: float, b: float, g: float, h: float) -> Array:
    z = np.asarray(z, dtype=float)
    tail = np.exp(0.5 * h * z * z)
    if abs(g) < 1e-8:
        return a + b * z * tail
    return a + b * (np.exp(g * z) - 1.0) / g * tail


def tukey_gh_ppf(p: float, a: float = 0.0, b: float = 1.0, g: float = 0.0, h: float = 0.0) -> float:
    """g-and-h quantile function."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if b <= 0.0 or h < 0.0:
        raise ValueError("b must be positive and h non-negative")
    return float(_transform(np.array([norm.ppf(p)]), a, b, g, h)[0])


def _z_of_x(x: float, a: float, b: float, g: float, h: float) -> float:
    # Monotone in z for h >= 0, b > 0; invert on a wide bracket.
    return float(brentq(lambda z: float(_transform(np.array([z]), a, b, g, h)[0]) - x, -40.0, 40.0))


def tukey_gh_cdf(x: Array, a: float = 0.0, b: float = 1.0, g: float = 0.0, h: float = 0.0) -> Array:
    """g-and-h CDF via numerical inversion of the monotone transform."""
    if b <= 0.0 or h < 0.0:
        raise ValueError("b must be positive and h non-negative")
    xs = np.atleast_1d(np.asarray(x, dtype=float))
    return np.array([norm.cdf(_z_of_x(float(v), a, b, g, h)) for v in xs])


def tukey_gh_pdf(x: Array, a: float = 0.0, b: float = 1.0, g: float = 0.0, h: float = 0.0) -> Array:
    """g-and-h density via the change-of-variables f(x) = phi(z) / (dx/dz)."""
    if b <= 0.0 or h < 0.0:
        raise ValueError("b must be positive and h non-negative")
    xs = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.empty(xs.size)
    for i, v in enumerate(xs):
        z = _z_of_x(float(v), a, b, g, h)
        tail = np.exp(0.5 * h * z * z)
        if abs(g) < 1e-8:
            dxdz = b * tail * (1.0 + h * z * z)
        else:
            term = (np.exp(g * z) - 1.0) / g
            dxdz = b * tail * (np.exp(g * z) + h * z * term)
        out[i] = norm.pdf(z) / abs(dxdz)
    return out


def tukey_gh_fit(x: Array, ps: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4)) -> dict[str, float]:
    """Hoaglin quantile fit; returns ``a`` (median), ``b``, ``g``, ``h``."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < 30 or not np.isfinite(arr).all():
        raise ValueError("x must be finite with >= 30 observations")
    a = float(np.median(arr))
    g_vals: list[float] = []
    zsq_half: list[float] = []
    log_corr: list[float] = []
    for pl in ps:
        if not 0.0 < pl < 0.5:
            raise ValueError("ps must lie in (0, 0.5)")
        pu = 1.0 - pl
        z = float(norm.ppf(pu))  # > 0
        x_lo, x_hi = (float(v) for v in np.quantile(arr, [pl, pu]))
        uhs = x_hi - a
        lhs = a - x_lo
        if uhs <= 0.0 or lhs <= 0.0:
            raise ValueError("degenerate spreads; cannot fit")
        g_p = (1.0 / z) * np.log(uhs / lhs)
        g_vals.append(g_p)
        full = uhs + lhs
        if abs(g_p) < 1e-8:
            corrected = full / (2.0 * z)
        else:
            corrected = full * g_p / (2.0 * np.sinh(g_p * z))
        log_corr.append(float(np.log(corrected)))
        zsq_half.append(0.5 * z * z)
    g = float(np.median(g_vals))
    # Regress log corrected spread on z^2/2: slope = h, intercept = ln B.
    design = np.column_stack([np.ones(len(zsq_half)), np.array(zsq_half)])
    coef, *_ = np.linalg.lstsq(design, np.array(log_corr), rcond=None)
    ln_b, h = float(coef[0]), float(coef[1])
    return {"a": a, "b": float(np.exp(ln_b)), "g": g, "h": max(h, 0.0)}
