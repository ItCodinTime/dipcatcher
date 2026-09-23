"""Multifractal detrended fluctuation analysis (Kantelhardt et al. 2002).

MF-DFA generalizes DFA to q-th order fluctuation functions:
  profile Y(i) = sum_{k<=i} (x_k - mean x); per scale s, fit a local
  polynomial of order m on each of 2 N_s segments (forward+backward),
  F_q(s) = [ (1/2N_s) sum_v F_v(s)^q ]^{1/q}  (q != 0; q=0 uses the
  geometric mean). Then F_q(s) ~ s^{h(q)}; mass exponent
  tau(q) = q h(q) - 1; singularity spectrum f(alpha) via Legendre
  transform alpha = dtau/dq, f = q alpha - tau.

Fail-closed: series too short for the largest scale, non-finite input,
degenerate (zero-variance) fluctuations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _segment_fluctuations(y_profile: Array, s: int, order: int) -> Array:
    """RMS detrended fluctuation per segment, forward + backward."""
    n = y_profile.size
    n_seg = n // s
    t = np.arange(s)
    f2 = []
    for direction in (0, 1):
        for v in range(n_seg):
            if direction == 0:
                seg = y_profile[v * s : (v + 1) * s]
            else:
                seg = y_profile[n - (v + 1) * s : n - v * s]
            coef = np.polyfit(t, seg, order)
            resid = seg - np.polyval(coef, t)
            f2.append(float(np.mean(resid * resid)))
    return np.asarray(f2)


def mfdfa(
    x: Array,
    scales: Array | None = None,
    q_list: Array | None = None,
    order: int = 1,
) -> dict[str, Array | float]:
    """MF-DFA fluctuation functions + mass exponent + singularity spectrum.

    ``scales``: segment sizes (default log-spaced 16..n/8, 10 points).
    ``q_list``: q grid (default -5..5 step 0.5).
    Returns dict: scales, q, fq (len(scales) x len(q)), h_q, tau_q,
    alpha, f_alpha, spectrum_width (alpha_max - alpha_min), h_at_q2.
    """
    xx = np.asarray(x, dtype=float).ravel()
    if xx.size < 64 or not np.isfinite(xx).all():
        raise ValueError("x must be finite, >= 64 obs")
    if float(xx.std()) == 0.0:
        raise ValueError("x must have nonzero variance")
    n = xx.size
    if scales is None:
        scales = np.unique(
            np.floor(np.logspace(np.log10(16), np.log10(max(32, n // 8)), 10)).astype(int)
        )
    sc = np.asarray(scales, dtype=int).ravel()
    if sc.size < 3 or sc.min() < 4 * (order + 1) or sc.max() > n // 4:
        raise ValueError("scales must satisfy 4(order+1) <= s <= n/4, >= 3 scales")
    if order < 0 or order > 3:
        raise ValueError("order must be 0..3")
    q = np.arange(-5.0, 5.5, 0.5) if q_list is None else np.asarray(q_list, dtype=float).ravel()
    if q.size < 3 or not np.isfinite(q).all():
        raise ValueError("q_list must be finite, >= 3 values")

    profile = np.cumsum(xx - xx.mean())
    fq = np.full((sc.size, q.size), np.nan)
    for i, s in enumerate(sc):
        f2 = _segment_fluctuations(profile, int(s), order)
        f2 = f2[f2 > 0]
        if f2.size < 2:
            continue
        for j, qv in enumerate(q):
            if abs(qv) < 1e-9:
                fq[i, j] = float(np.exp(0.5 * np.mean(np.log(f2))))
            else:
                fq[i, j] = float(np.mean(f2 ** (qv / 2.0)) ** (1.0 / qv))
    if np.isnan(fq).all():
        raise ValueError("all segments degenerate")

    log_s = np.log(sc.astype(float))
    h_q = np.full(q.size, np.nan)
    for j in range(q.size):
        col = fq[:, j]
        ok = np.isfinite(col) & (col > 0)
        if ok.sum() < 3:
            continue
        h_q[j] = np.polyfit(log_s[ok], np.log(col[ok]), 1)[0]
    if np.isnan(h_q).all():
        raise ValueError("no estimable h(q)")
    tau = q * h_q - 1.0
    ok = np.isfinite(h_q)
    # Legendre transform of tau(q) -> singularity spectrum
    alpha = np.gradient(tau[ok], q[ok])
    f_alpha = q[ok] * alpha - tau[ok]
    i2 = int(np.argmin(np.abs(q - 2.0)))
    return {
        "scales": sc.astype(float),
        "q": q,
        "fq": fq,
        "h_q": h_q,
        "tau_q": tau,
        "alpha": np.asarray(alpha, dtype=float),
        "f_alpha": np.asarray(f_alpha, dtype=float),
        "spectrum_width": float(np.nanmax(alpha) - np.nanmin(alpha)),
        "h_at_q2": float(h_q[i2]),
    }
