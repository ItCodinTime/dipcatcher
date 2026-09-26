"""Carr-Madan (1999) FFT option pricing from a characteristic function.

For a risk-neutral log-price characteristic function ``phi(u) = E[e^{i u ln S_T}]``
the damped call transform is

    psi(v) = e^{-rT} phi(v - (alpha+1) i)
             / (alpha^2 + alpha - v^2 + i (2 alpha + 1) v),

and call prices follow from ``C(k) = e^{-alpha k}/pi Re integral_0^inf e^{-i v k}
psi(v) dv``, computed with the FFT over a grid of log-strikes.  The damping
parameter ``alpha > 0`` makes the transform integrable.

Reference: P. Carr, D. Madan (1999), "Option valuation using the fast Fourier
transform", Journal of Computational Finance.  Fail-closed on invalid inputs.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
CharFn = Callable[[np.ndarray], np.ndarray]


def bs_char_fn(s0: float, r: float, t: float, sigma: float) -> CharFn:
    """Black-Scholes risk-neutral characteristic function of ln S_T."""

    def phi(u: np.ndarray) -> np.ndarray:
        mu = np.log(s0) + (r - 0.5 * sigma**2) * t
        return np.asarray(np.exp(1j * u * mu - 0.5 * sigma**2 * t * u**2), dtype=complex)

    return phi


def carr_madan_call(
    char_fn: CharFn,
    r: float,
    t: float,
    strikes: Array,
    alpha: float = 1.5,
    n: int = 4096,
    eta: float = 0.25,
) -> Array:
    """European call prices at ``strikes`` via the Carr-Madan FFT."""
    if t <= 0.0 or alpha <= 0.0 or n <= 0 or eta <= 0.0:
        raise ValueError("require t > 0, alpha > 0, n > 0, eta > 0")
    ks = np.asarray(strikes, dtype=float).ravel()
    if ks.size == 0 or (ks <= 0).any() or not np.isfinite(ks).all():
        raise ValueError("strikes must be positive and finite")
    lam = 2.0 * np.pi / (n * eta)  # log-strike spacing
    b = n * lam / 2.0
    v = np.arange(n) * eta
    u = v - (alpha + 1.0) * 1j
    denom = alpha**2 + alpha - v**2 + 1j * (2.0 * alpha + 1.0) * v
    psi = np.exp(-r * t) * char_fn(u) / denom
    # Simpson weights for accuracy.
    simpson = (3.0 + (-1.0) ** np.arange(1, n + 1) - np.where(np.arange(n) == 0, 1.0, 0.0)) / 3.0
    fft_input = np.exp(1j * b * v) * psi * eta * simpson
    fft_vals = np.fft.fft(fft_input).real
    log_strikes = -b + lam * np.arange(n)
    call_grid = np.exp(-alpha * log_strikes) / np.pi * fft_vals
    grid_k = np.exp(log_strikes)
    return np.asarray(np.interp(ks, grid_k, call_grid), dtype=float)
