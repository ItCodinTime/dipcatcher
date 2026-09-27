"""CGMY (tempered stable) Levy process.

Carr, Geman, Madan & Yor (2002) generalise the Variance-Gamma process with a
tail index ``Y``.  Its characteristic exponent (for unit time) is

    psi(u) = C Gamma(-Y) [ (M - i u)^Y - M^Y + (G + i u)^Y - G^Y ],

with ``C > 0`` controlling activity, ``G, M > 0`` the down/up tempering and
``Y < 2`` the fine structure (``Y = 0`` recovers Variance-Gamma).  Cumulants are
closed form,

    c_k = C Gamma(k - Y) [ M^{Y-k} + (-1)^k G^{Y-k} ],

and the density is obtained by numerical Fourier inversion of ``exp(psi)``.

Reference: P. Carr, H. Geman, D. Madan, M. Yor (2002), "The fine structure of
asset returns", Journal of Business.  Fail-closed on invalid parameters.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.special import gamma as gamma_fn

Array = NDArray[np.float64]


def _check(c: float, g: float, m: float, y: float) -> None:
    if c <= 0.0 or g <= 0.0 or m <= 0.0:
        raise ValueError("require C, G, M > 0")
    if y >= 2.0 or y == 1.0 or y == 0.0:
        raise ValueError("require Y < 2 and Y not in {0, 1} (use VG for Y=0)")


def cgmy_char_exponent(u: np.ndarray, c: float, g: float, m: float, y: float) -> np.ndarray:
    """Characteristic exponent psi(u) with phi(u) = exp(psi(u))."""
    _check(c, g, m, y)
    return np.asarray(c * gamma_fn(-y) * ((m - 1j * u) ** y - m**y + (g + 1j * u) ** y - g**y))


def cgmy_cumulants(c: float, g: float, m: float, y: float) -> dict[str, float]:
    """First four cumulants and the standardised skew/kurtosis."""
    _check(c, g, m, y)
    c1 = c * gamma_fn(1.0 - y) * (m ** (y - 1.0) - g ** (y - 1.0))
    c2 = c * gamma_fn(2.0 - y) * (m ** (y - 2.0) + g ** (y - 2.0))
    c3 = c * gamma_fn(3.0 - y) * (m ** (y - 3.0) - g ** (y - 3.0))
    c4 = c * gamma_fn(4.0 - y) * (m ** (y - 4.0) + g ** (y - 4.0))
    return {
        "mean": float(c1),
        "var": float(c2),
        "skew": float(c3 / c2**1.5),
        "exkurt": float(c4 / c2**2),
        "c1": float(c1),
        "c2": float(c2),
        "c3": float(c3),
        "c4": float(c4),
    }


def cgmy_pdf(x: Array, c: float, g: float, m: float, y: float) -> Array:
    """CGMY density via numerical Fourier inversion of exp(psi)."""
    _check(c, g, m, y)
    xs = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.empty(xs.size)
    for i, xv in enumerate(xs):

        def integrand(u: float, xv: float = xv) -> float:
            phi = np.exp(cgmy_char_exponent(np.array([u]), c, g, m, y)[0])
            return float((np.exp(-1j * u * xv) * phi).real)

        val, _ = quad(integrand, 0.0, 200.0, limit=200)
        out[i] = max(val / np.pi, 0.0)
    return out
