"""Spectral-density estimation and cross-spectral analysis.

References:
- Welch (1967): averaged periodogram over overlapped segments.
- Daniell (1946): smoothed periodogram (moving-average kernel).
- Thomson (1982): multitaper estimate with DPSS (Slepian) tapers.
- Priestley (1981): coherence and cross-spectrum conventions.
- Parzen (1957): spectral windows.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import windows

Array = NDArray[np.float64]


def _v(x: Array, n: int = 32) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size < n or not np.all(np.isfinite(v)):
        raise ValueError(f"series must be finite with length >= {n}")
    if v.std() == 0:
        raise ValueError("degenerate (constant) series")
    return v


def periodogram_daniell(series: Array, m: int = 5) -> dict[str, Array]:
    """Daniell (1946) smoothed periodogram — moving average over
    2m+1 adjacent ordinates. Returns frequencies in cycles/sample."""
    v = _v(series)
    n = v.size
    if not (1 <= m < n // 8):
        raise ValueError("m must be in [1, n/8)")
    vc = v - v.mean()
    F = np.abs(np.fft.rfft(vc)) ** 2 / n
    freqs = np.fft.rfftfreq(n)
    kern = np.ones(2 * m + 1) / (2 * m + 1)
    sm = np.convolve(F, kern, mode="same")
    return {"freqs": freqs, "density": sm, "raw": F}


def periodogram_welch(
    series: Array, seg_len: int = 128, overlap: float = 0.5
) -> dict[str, Array | float]:
    """Welch (1967) averaged periodogram with Hann-tapered segments."""
    v = _v(series)
    n = v.size
    if not (16 <= seg_len <= n):
        raise ValueError("seg_len must be in [16, n]")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1)")
    step = max(1, int(seg_len * (1.0 - overlap)))
    win = windows.hann(seg_len, sym=False)
    win2 = float(win @ win)
    acc = np.zeros(seg_len // 2 + 1)
    cnt = 0
    for s in range(0, n - seg_len + 1, step):
        seg = (v[s : s + seg_len] - v[s : s + seg_len].mean()) * win
        acc += np.abs(np.fft.rfft(seg)) ** 2 / win2
        cnt += 1
    if cnt == 0:
        raise ValueError("no complete segments")
    return {"freqs": np.fft.rfftfreq(seg_len), "density": acc / cnt, "n_seg": float(cnt)}


def _dpss_tapers(n: int, nw: float, k: int) -> Array:
    """Slepian (DPSS) tapers via scipy."""
    return np.asarray(windows.dpss(n, nw, Kmax=k, sym=False), dtype=float)


def periodogram_multitaper(
    series: Array, nw: float = 3.0, k_tapers: int | None = None
) -> dict[str, Array | float]:
    """Thomson (1982) multitaper PSD with DPSS tapers.

    ``nw`` is the time-halfbandwidth product; default 3 gives ~5 tapers.
    Averages the direct (adaptively weighted) eigenspectra — equal-weight
    version (Thomson's simplest form)."""
    v = _v(series)
    n = v.size
    if not (1.0 <= nw <= 8.0):
        raise ValueError("nw must be in [1, 8]")
    k = int(2 * nw - 1) if k_tapers is None else int(k_tapers)
    if not (1 <= k <= 2 * nw + 1):
        raise ValueError("k_tapers out of range")
    vc = v - v.mean()
    tapers = _dpss_tapers(n, nw, k)
    spec = np.zeros(n // 2 + 1)
    for tap in tapers:
        F = np.fft.rfft(vc * tap)
        spec += np.abs(F) ** 2
    spec /= k
    return {"freqs": np.fft.rfftfreq(n), "density": spec, "n_tapers": float(k)}


def coherence(x: Array, y: Array, seg_len: int = 64) -> dict[str, Array | float]:
    """Priestley-style magnitude-squared coherence via Welch segments.

    ``C_xy(f) = |S_xy(f)|^2 / (S_xx S_yy)`` in [0, 1]."""
    a = _v(x)
    b = _v(y)
    if a.size != b.size:
        raise ValueError("series must share length")
    n = a.size
    if not (16 <= seg_len <= n):
        raise ValueError("seg_len must be in [16, n]")
    step = seg_len // 2
    win = windows.hann(seg_len, sym=False)
    win2 = float(win @ win)
    Sxx = np.zeros(seg_len // 2 + 1)
    Syy = np.zeros(seg_len // 2 + 1)
    Sxy = np.zeros(seg_len // 2 + 1, dtype=complex)
    cnt = 0
    for s in range(0, n - seg_len + 1, step):
        sa = (a[s : s + seg_len] - a[s : s + seg_len].mean()) * win
        sb = (b[s : s + seg_len] - b[s : s + seg_len].mean()) * win
        Fa = np.fft.rfft(sa)
        Fb = np.fft.rfft(sb)
        Sxx += np.abs(Fa) ** 2 / win2
        Syy += np.abs(Fb) ** 2 / win2
        Sxy += Fa * np.conj(Fb) / win2
        cnt += 1
    if cnt == 0:
        raise ValueError("no complete segments")
    coh = np.abs(Sxy) ** 2 / np.maximum(Sxx * Syy, 1e-300)
    return {"freqs": np.fft.rfftfreq(seg_len), "coherence": np.clip(coh, 0, 1), "n_seg": float(cnt)}


def band_power(freqs: Array, density: Array, lo: float, hi: float) -> float:
    """Integrated band power: trapezoid integral of density over [lo, hi]."""
    f = np.asarray(freqs, dtype=float).reshape(-1)
    d = np.asarray(density, dtype=float).reshape(-1)
    if f.size != d.size or f.size < 4 or not np.all(np.isfinite(d)):
        raise ValueError("freqs/density must be finite and aligned")
    if not (0 <= lo < hi <= f.max()):
        raise ValueError("band must lie within the frequency grid")
    mask = (f >= lo) & (f <= hi)
    if mask.sum() < 2:
        raise ValueError("band contains too few frequency bins")
    return float(np.trapezoid(d[mask], f[mask]))


def dominant_frequency(freqs: Array, density: Array, lo: float = 0.0) -> float:
    """Frequency of the spectral peak above ``lo``."""
    f = np.asarray(freqs, dtype=float).reshape(-1)
    d = np.asarray(density, dtype=float).reshape(-1)
    if f.size != d.size or f.size < 4 or not np.all(np.isfinite(d)):
        raise ValueError("freqs/density must be finite and aligned")
    mask = f > lo
    if mask.sum() == 0:
        raise ValueError("no frequencies above lo")
    idx = np.argmax(d[mask])
    return float(f[mask][idx])
