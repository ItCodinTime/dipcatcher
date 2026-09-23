"""Signal decomposition: SSA and EMD/Hilbert–Huang.

Nonparametric decomposition of a series into trend/oscillation/noise
components, plus an SSA linear-recurrent-formula forecaster.

References:
- Broomhead, King (1986); Vautard, Yiou, Ghil (1992); Golyandina,
    Nekrutkin, Zhigljavsky (2001) — singular spectrum analysis.
- Elsner, Tsonis (1996) — SSA applications in finance.
- Huang et al. (1998) — empirical mode decomposition (sifting).
- Huang et al. (2003) — Hilbert spectral analysis / instantaneous freq.
- Wu, Huang (2009) — EEMD noise-assisted ensemble (optional averaging).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import interpolate, signal

Array = NDArray[np.float64]


def _v(x: Array, n: int = 8) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size < n or not np.all(np.isfinite(v)):
        raise ValueError(f"series must be finite with length >= {n}")
    return v


def ssa_decompose(y: Array, window: int = 40) -> dict[str, Array]:
    """Broomhead–King SSA decomposition.

    Embeds the centered series into a (L x K) trajectory matrix, takes the
    SVD, and diagonal-averages each rank-1 component back to a series.
    Returns the per-component reconstructed series (n, L) and eigenvalue
    spectrum for grouping/selection.
    """
    v = _v(y)
    n = v.size
    if window < 2 or window > n // 2:
        raise ValueError("window must be in [2, n/2]")
    L = window
    K = n - L + 1
    vc = v - v.mean()
    # Trajectory matrix: column j is v[j:j+L].
    traj = np.lib.stride_tricks.sliding_window_view(vc, L).T  # (L, K)
    u, s, vt = np.linalg.svd(traj, full_matrices=False)
    recon = np.zeros((n, L))
    for i in range(L):
        comp = np.outer(u[:, i], vt[i]) * s[i]
        recon[:, i] = _diagonal_average(comp, n)
    eig = s * s / K
    return {
        "components": recon,
        "eigvals": eig,
        "share": eig / eig.sum(),
        "window": np.array([float(L)]),
    }


def _diagonal_average(m: NDArray[np.float64], n: int) -> Array:
    """Hankelization: average each anti-diagonal of the (L, K) matrix."""
    L, K = m.shape
    out = np.zeros(n)
    for t in range(n):
        lo = max(0, t - K + 1)
        hi = min(L - 1, t)
        vals = [m[i, t - i] for i in range(lo, hi + 1)]
        out[t] = float(np.mean(vals))
    return out


def ssa_forecast(y: Array, window: int = 40, groups: int = 4, steps: int = 5) -> dict[str, Array]:
    """SSA recurrent forecasting (Golyandina et al. 2001, 'vector' R-forecast).

    Uses the first ``groups`` eigentriples: builds the linear recurrence
    from the vertical eigenvector coefficients and rolls forward.
    """
    v = _v(y)
    dec = ssa_decompose(v, window=window)
    L = int(dec["window"][0])
    comp = dec["components"][:, :groups].sum(axis=1)
    # Recurrence coefficients from the last eigenvectors (verticality).
    vc = v - v.mean()
    traj = np.lib.stride_tricks.sliding_window_view(vc, L).T
    u, s, vt = np.linalg.svd(traj, full_matrices=False)
    U = u[:, :groups]
    # R coefficients: a = sum_i U_i[:-1] * U_i[-1] / (1 - sum_i U_i[-1]^2)
    denom = 1.0 - float(np.sum(U[-1, :] ** 2))
    if denom <= 1e-8:
        raise ValueError("vertical eigenvectors degenerate")
    a = (U[:-1, :] @ U[-1, :]) / denom  # (L-1,)
    mean = float(v.mean())
    tail = (comp[-(L - 1) :] - mean).tolist()
    fc = []
    for _ in range(steps):
        nxt = float(np.dot(a, tail[-(L - 1) :]))
        fc.append(nxt + mean)
        tail.append(nxt)
    return {
        "forecast": np.asarray(fc),
        "reconstructed": comp + mean,
        "recurrence": a,
    }


def emd(y: Array, max_imfs: int = 8, max_sift: int = 50, sd_tol: float = 0.3) -> dict[str, Array]:
    """Huang et al. (1998) empirical mode decomposition.

    Sifts the series into intrinsic mode functions via cubic-spline
    envelopes of extrema. Stops when the residue is monotone or the sift
    SD criterion is met. Returns IMF matrix (n, n_imf) and residue.
    """
    v = _v(y)
    residue = v.copy()
    imfs: list[Array] = []
    for _ in range(max_imfs):
        if _is_monotone(residue):
            break
        h = residue.copy()
        for _ in range(max_sift):
            env = _envelopes(h)
            if env is None:
                break
            mean = 0.5 * (env[0] + env[1])
            h_new = h - mean
            sd = float(np.sum((h - h_new) ** 2) / np.maximum(np.sum(h * h), 1e-12))
            h = h_new
            if sd < sd_tol:
                break
        imfs.append(h)
        residue = residue - h
    mat = np.column_stack(imfs) if imfs else np.zeros((v.size, 0))
    return {"imfs": mat, "residue": residue, "n_imf": np.array([float(mat.shape[1])])}


def _is_monotone(v: Array) -> bool:
    d = np.diff(v)
    return bool(np.all(d >= 0) or np.all(d <= 0))


def _envelopes(v: Array) -> tuple[Array, Array] | None:
    """Upper/lower cubic envelopes through extrema; None if <2 extrema."""
    n = v.size
    maxima = signal.argrelextrema(v, np.greater)[0]
    minima = signal.argrelextrema(v, np.less)[0]
    if maxima.size < 2 or minima.size < 2:
        return None
    xs = np.arange(n)
    # Include endpoints in envelope interpolation.
    xs_max = np.concatenate([[0], maxima, [n - 1]])
    xs_min = np.concatenate([[0], minima, [n - 1]])
    upper = interpolate.CubicSpline(xs_max, v[xs_max])(xs)
    lower = interpolate.CubicSpline(xs_min, v[xs_min])(xs)
    return upper, lower


def hilbert_spectrum(y: Array) -> dict[str, Array]:
    """Hilbert–Huang instantaneous amplitude/frequency of the first IMF.

    Runs EMD, takes the analytic signal of IMF 1, returns instantaneous
    amplitude, phase, and frequency (cycles/sample).
    """
    v = _v(y)
    dec = emd(v)
    if dec["imfs"].shape[1] == 0:
        raise ValueError("no IMF extracted — series may be monotone")
    imf1 = dec["imfs"][:, 0]
    analytic = signal.hilbert(imf1)
    amp = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    freq = np.diff(phase) / (2.0 * np.pi)
    return {
        "imf": imf1,
        "amplitude": amp,
        "phase": phase,
        "inst_freq": np.concatenate([[np.nan], freq]),
    }


def robust_trend(y: Array, window: int = 21) -> dict[str, Array]:
    """Median-filter trend + residual — Hodrick-Prescott-robust trend proxy.

    Uses a rolling median (Hodges/Lehmann robust smoother) — the standard
    robust alternative before bandpass-style detrending.
    """
    v = _v(y)
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be odd and >= 3")
    trend = signal.medfilt(v, kernel_size=window)
    return {"trend": trend, "detrended": v - trend}
