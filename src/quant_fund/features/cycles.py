"""Cycle and spectral filters (Ehlers, Goertzel, FFT, Hilbert).

References:
- Goertzel (1958). An algorithm for the evaluation of finite trigonometric
  series — single-bin DFT power.
- Ehlers (2013). *Cycle Analytics for Traders* — SuperSmoother, roofing
  filter, bandpass filter, Hilbert instantaneous frequency.
- Bracewell (1999). *The Fourier Transform and Its Applications* — FFT
  dominant-cycle extraction.
- Ehlers FIR quadrature (in-phase lagged 3 bars, 7-tap detrender). Bar ``t``
  does not see ``t+1``. This is not ``scipy.signal.hilbert``.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _as_vec(x: Array, name: str = "x", min_n: int = 16) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size < min_n or not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be a finite vector of length >= {min_n}")
    return v


def goertzel_power(x: Array, period: int) -> float:
    """Goertzel (1958) normalized power at the given cycle period.

    ``period`` in bars (>= 4).  Returns spectral power at
    ``f = 1/period`` cycles/bar.
    """
    v = _as_vec(x)
    if isinstance(period, bool) or not isinstance(period, int) or period < 4:
        raise ValueError("period must be an integer >= 4")
    if period > v.size // 2:
        raise ValueError("period must be <= len(x)/2")
    n = v.size
    omega = 2.0 * math.pi / period
    coeff = 2.0 * math.cos(omega)
    s_prev = 0.0
    s_prev2 = 0.0
    for xn in v - v.mean():
        s = xn + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    power = s_prev2**2 + s_prev**2 - coeff * s_prev * s_prev2
    return float(power / (n * n))


def cycle_periodogram(x: Array, periods: Array | None = None) -> tuple[Array, Array]:
    """Goertzel power spectrum over a grid of periods (bars)."""
    v = _as_vec(x)
    ps = (
        np.arange(6, v.size // 2 + 1, dtype=np.intp)
        if periods is None
        else np.asarray(periods, dtype=np.intp).reshape(-1)
    )
    if ps.size < 2 or np.any(ps < 4) or np.any(ps > v.size // 2):
        raise ValueError("periods must be integers in [4, len(x)/2]")
    power = np.array([goertzel_power(v, int(p)) for p in ps])
    return ps.astype(float), power


def dominant_cycle_fft(x: Array, min_period: int = 6, max_period: int = 64) -> dict[str, float]:
    """Dominant cycle period from the FFT magnitude spectrum.

    Returns ``{period, power_fraction, snr_vs_mean}`` restricted to
    ``[min_period, max_period]`` bars.
    """
    v = _as_vec(x)
    if min_period < 4 or max_period > v.size // 2 or min_period >= max_period:
        raise ValueError("require 4 <= min_period < max_period <= len(x)/2")
    n = v.size
    d = v - v.mean()
    spec = np.abs(np.fft.rfft(d))
    freqs = np.fft.rfftfreq(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
    mask = (periods >= min_period) & (periods <= max_period)
    if not np.any(mask):
        raise ValueError("no FFT bin falls inside the requested band")
    idx = np.flatnonzero(mask)
    best = idx[int(np.argmax(spec[idx]))]
    band_power = spec[idx] ** 2
    frac = float((spec[best] ** 2) / np.sum(spec[1:] ** 2)) if spec[1:].sum() > 0 else 0.0
    snr = float((spec[best] ** 2) / band_power.mean()) if band_power.size else 0.0
    return {
        "period": float(periods[best]),
        "power_fraction": frac,
        "snr_vs_mean": snr,
    }


def _ehlers_quadrature(x: Array) -> tuple[Array, Array]:
    """Causal Ehlers quadrature. Every tap is a lag; no FFT of the future."""
    v = np.asarray(x, dtype=float)
    n = v.size
    smooth = np.full(n, np.nan)
    for i in range(3, n):
        smooth[i] = (4.0 * v[i] + 3.0 * v[i - 1] + 2.0 * v[i - 2] + v[i - 3]) / 10.0
    in_phase = np.full(n, np.nan)
    quad = np.full(n, np.nan)
    for i in range(6, n):
        window = smooth[i - 6 : i + 1]
        if not np.isfinite(window).all():
            continue
        quad[i] = (
            0.0962 * smooth[i]
            + 0.5769 * smooth[i - 2]
            - 0.5769 * smooth[i - 4]
            - 0.0962 * smooth[i - 6]
        )
        in_phase[i] = smooth[i - 3]
    return in_phase, quad


def hilbert_instantaneous_frequency(x: Array, smooth: int = 5) -> Array:
    """Instantaneous cycle period in bars. Causal Ehlers FIR, not an FFT Hilbert.

    Phase change at bar ``t`` uses the quadrature pair at ``t`` and ``t-1`` only.
    ``smooth`` is a trailing mean. Warmup stays NaN.
    """
    v = _as_vec(x)
    if isinstance(smooth, bool) or not isinstance(smooth, int) or smooth < 1:
        raise ValueError("smooth must be a positive integer")
    in_phase, quad = _ehlers_quadrature(v)
    raw = np.full(v.size, np.nan)
    prev: float | None = None
    for i in range(v.size):
        if not (np.isfinite(in_phase[i]) and np.isfinite(quad[i])):
            prev = None
            continue
        phase = math.atan2(float(quad[i]), float(in_phase[i]))
        if prev is None:
            prev = phase
            continue
        delta = phase - prev
        delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
        prev = phase
        if abs(delta) > 1e-8:
            raw[i] = (2.0 * math.pi) / abs(delta)
    if smooth == 1:
        return raw
    out = np.full(v.size, np.nan)
    for i in range(v.size):
        seg = raw[i - smooth + 1 : i + 1]
        if seg.size == smooth and np.isfinite(seg).all():
            out[i] = float(seg.mean())
    return out


def supersmoother(x: Array, period: float = 10.0) -> Array:
    """Ehlers 2-pole SuperSmoother.

    ``a1 = exp(-sqrt(2) pi / period)``, coefficients per Ehlers (2013).
    """
    v = _as_vec(x)
    if not np.isfinite(period) or period < 2.0:
        raise ValueError("period must be >= 2")
    a1 = math.exp(-math.sqrt(2.0) * math.pi / period)
    b1 = 2.0 * a1 * math.cos(math.sqrt(2.0) * math.pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1.0 - c2 - c3
    out = np.zeros(v.size)
    out[0] = v[0]
    out[1] = v[1]
    for i in range(2, v.size):
        out[i] = c1 * (v[i] + v[i - 1]) / 2.0 + c2 * out[i - 1] + c3 * out[i - 2]
    return out


def roofing_filter(x: Array, hp_period: float = 48.0, lp_period: float = 10.0) -> Array:
    """Ehlers roofing filter = 2-pole highpass + SuperSmoother lowpass."""
    v = _as_vec(x)
    if not np.isfinite(hp_period) or hp_period <= 2.0:
        raise ValueError("hp_period must be > 2")
    if not np.isfinite(lp_period) or lp_period < 2.0:
        raise ValueError("lp_period must be >= 2")
    a1 = math.exp(-math.sqrt(2.0) * math.pi / hp_period)
    b1 = 2.0 * a1 * math.cos(math.sqrt(2.0) * math.pi / hp_period)
    c2 = b1
    c3 = -a1 * a1
    c1 = (1.0 + c2) / 2.0
    hp = np.zeros(v.size)
    for i in range(2, v.size):
        hp[i] = c1 * (v[i] - 2.0 * v[i - 1] + v[i - 2]) + c2 * hp[i - 1] + c3 * hp[i - 2]
    return supersmoother(hp, lp_period)


def bandpass_filter(
    x: Array,
    period: float = 20.0,
    bandwidth: float = 0.3,
) -> Array:
    """Ehlers bandpass filter centered on ``period`` (bars).

    ``bandwidth`` is the fractional octave width (~0.3 default).
    """
    v = _as_vec(x)
    if not np.isfinite(period) or period < 2.0:
        raise ValueError("period must be >= 2")
    if not np.isfinite(bandwidth) or not (0.0 < bandwidth < 1.0):
        raise ValueError("bandwidth must be in (0, 1)")
    delta = bandwidth
    beta = math.cos(2.0 * math.pi / period)
    gamma = 1.0 / math.cos(2.0 * math.pi * delta / period)
    alpha = gamma - math.sqrt(max(gamma * gamma - 1.0, 0.0))
    out = np.zeros(v.size)
    for i in range(3, v.size):
        out[i] = (
            0.5 * (1.0 - alpha) * (v[i] - v[i - 2])
            + beta * (1.0 + alpha) * out[i - 1]
            - alpha * out[i - 2]
        )
    return out


def hilbert_transform_indicator(x: Array) -> dict[str, Array]:
    """Causal in-phase / quadrature pair (Ehlers FIR). Not ``scipy.signal.hilbert``."""
    v = _as_vec(x)
    in_phase, quad = _ehlers_quadrature(v)
    amplitude = np.full(v.size, np.nan)
    phase = np.full(v.size, np.nan)
    ok = np.isfinite(in_phase) & np.isfinite(quad)
    amplitude[ok] = np.hypot(in_phase[ok], quad[ok])
    phase[ok] = np.arctan2(quad[ok], in_phase[ok])
    return {
        "in_phase": in_phase,
        "quadrature": quad,
        "amplitude": amplitude,
        "phase": phase,
    }
