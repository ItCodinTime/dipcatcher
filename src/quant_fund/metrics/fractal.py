"""Fractal scaling diagnostics: Hurst exponent and fractal dimension.

Long-memory / roughness probes used in volatility and regime research.
Estimators are log-log regressions over dyadic scaling ranges — sensitive to
window choice, so results are research diagnostics with the fitted range
reported back.

References:
- Hurst (1951). Long-term storage capacity of reservoirs. *Trans. ASCE* 116.
- Mandelbrot, Wallis (1969). Robustness of the rescaled range R/S.
- Peng et al. (1994). Mosaic organization of DNA nucleotides — DFA.
  *Phys. Rev. E* 49.
- Kantz, Schreiber (1997). *Nonlinear Time Series Analysis* — window scaling.
- Katz (1988). Fractals and the analysis of waveforms. *Comput. Biol. Med.*
- Higuchi (1988). Approach to an irregular time series on the basis of the
  fractal theory. *Physica D* 31.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _as_vector(x: Array, name: str = "x", *, min_obs: int = 32) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size < min_obs:
        raise ValueError(f"{name} must contain at least {min_obs} finite observations")
    return v


def _dyadic_windows(n: int, min_win: int, max_win: int) -> list[int]:
    lo = max(4, min_win)
    hi = min(max_win, n // 2)
    sizes = sorted({int(round(2.0**e)) for e in np.arange(np.log2(lo), np.log2(hi) + 1e-9, 0.5)})
    return [s for s in sizes if lo <= s <= hi]


def hurst_rs(x: Array, min_win: int = 8, max_win: int | None = None) -> dict[str, float]:
    """Rescaled-range (R/S) Hurst estimate over dyadic window sizes.

    Slope of ``log(mean R/S)`` vs ``log(n)``.  H=0.5 iid; H>0.5 persistent;
    H<0.5 anti-persistent.  Known upward bias for small samples — diagnostic.
    """
    v = _as_vector(x)
    hi = v.size // 2 if max_win is None else max_win
    sizes = _dyadic_windows(v.size, min_win, hi)
    if len(sizes) < 3:
        raise ValueError("series too short for R/S scaling")
    log_n, log_rs = [], []
    for w in sizes:
        n_blocks = v.size // w
        rs_vals = []
        for i in range(n_blocks):
            seg = v[i * w : (i + 1) * w] - v[i * w : (i + 1) * w].mean()
            cum = np.cumsum(seg)
            r = float(cum.max() - cum.min())
            s = float(seg.std(ddof=1))
            if s > 0.0:
                rs_vals.append(r / s)
        if rs_vals:
            log_n.append(np.log(w))
            log_rs.append(np.log(np.mean(rs_vals)))
    if len(log_n) < 3:
        raise ValueError("insufficient scaling range")
    slope, *_ = np.polyfit(log_n, log_rs, 1)
    return {"hurst": float(slope), "n_scales": float(len(log_n))}


def dfa_hurst(
    x: Array, min_win: int = 8, max_win: int | None = None, poly_order: int = 1
) -> dict[str, float]:
    """Peng et al. (1994) detrended fluctuation analysis on the integrated series.

    RMS residual of per-window polynomial detrending vs window size; slope is
    the DFA scaling exponent (H for fractional Gaussian noise, alpha for fBm).
    """
    v = _as_vector(x)
    hi = v.size // 4 if max_win is None else max_win
    sizes = _dyadic_windows(v.size, min_win, hi)
    if len(sizes) < 3:
        raise ValueError("series too short for DFA scaling")
    if isinstance(poly_order, bool) or not isinstance(poly_order, int) or poly_order < 0:
        raise ValueError("poly_order must be a non-negative integer")
    profile = np.cumsum(v - v.mean())
    log_n, log_f = [], []
    for w in sizes:
        n_blocks = profile.size // w
        rms = []
        tt = np.arange(w)
        for i in range(n_blocks):
            seg = profile[i * w : (i + 1) * w]
            coef = np.polyfit(tt, seg, poly_order)
            resid = seg - np.polyval(coef, tt)
            rms.append(float(np.mean(resid**2)))
        f = float(np.sqrt(np.mean(rms)))
        if f > 0.0:
            log_n.append(np.log(w))
            log_f.append(np.log(f))
    if len(log_n) < 3:
        raise ValueError("insufficient scaling range")
    slope, *_ = np.polyfit(log_n, log_f, 1)
    return {"hurst": float(slope), "n_scales": float(len(log_n))}


def katz_fd(x: Array) -> float:
    """Katz (1988) fractal dimension ``log10(n) / (log10(n) + log10(d/L))``
    where ``L`` is the path length and ``d`` the max distance from the start."""
    v = _as_vector(x, min_obs=16)
    diffs = np.abs(np.diff(v))
    length = float(diffs.sum())
    d = float(np.max(np.abs(v - v[0])))
    n = float(v.size)
    if length <= 0.0 or d <= 0.0:
        raise ValueError("katz_fd requires a non-constant series")
    return float(np.log10(n) / (np.log10(n) + np.log10(d / length)))


def higuchi_fd(x: Array, kmax: int = 8) -> float:
    """Higuchi (1988) fractal dimension from the curve-length scaling law.

    ``L(k) ~ k^-FD``; slope of ``log(L(k))`` vs ``log(1/k)``.
    """
    v = _as_vector(x, min_obs=4 * max(2, kmax))
    if isinstance(kmax, bool) or not isinstance(kmax, int) or kmax < 2:
        raise ValueError("kmax must be an integer >= 2")
    n = v.size
    lk, xs = [], []
    for k in range(1, kmax + 1):
        lengths = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if idx.size < 2:
                continue
            scale = (n - 1.0) / ((idx.size - 1) * k)
            lengths.append(float(np.sum(np.abs(np.diff(v[idx]))) * scale))
        if lengths:
            lk.append(np.log(np.mean(lengths)))
            xs.append(np.log(1.0 / k))
    if len(lk) < 3:
        raise ValueError("insufficient scaling range")
    slope, *_ = np.polyfit(xs, lk, 1)
    return float(slope)


def sevcik_fd(x: Array) -> float:
    """Sevcik (1998) fractal dimension on the unit-normalized curve.

    ``FD = 1 + ln(L) / ln(2(n-1))`` with ``L`` the length of the series mapped
    to the unit square — a fast single-scale complement to Katz/Higuchi.
    """
    v = _as_vector(x, min_obs=16)
    rng = float(v.max() - v.min())
    if rng <= 0.0:
        raise ValueError("sevcik_fd requires a non-constant series")
    y = (v - v.min()) / rng
    t = np.arange(v.size) / (v.size - 1.0)
    length = float(np.sum(np.sqrt(np.diff(t) ** 2 + np.diff(y) ** 2)))
    return float(1.0 + np.log(length) / np.log(2.0 * (v.size - 1.0)))


def roughness_battery(x: Array) -> dict[str, float]:
    """Convenience bundle: Hurst (R/S + DFA) and FD (Katz, Higuchi, Sevcik)."""
    v = _as_vector(x)
    out: dict[str, float] = {}
    for name, fn in (
        ("hurst_rs", lambda: hurst_rs(v)["hurst"]),
        ("hurst_dfa", lambda: dfa_hurst(v)["hurst"]),
        ("katz_fd", lambda: katz_fd(v)),
        ("higuchi_fd", lambda: higuchi_fd(v)),
        ("sevcik_fd", lambda: sevcik_fd(v)),
    ):
        try:
            out[name] = float(fn())
        except ValueError:
            out[name] = float("nan")
    return out
