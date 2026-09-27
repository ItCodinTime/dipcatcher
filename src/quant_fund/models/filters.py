"""Trend/cycle decomposition filters.

Classical macro/finance signal extraction: Hodrick–Prescott, Baxter–King
bandpass, Christiano–Fitzgerald asymmetric bandpass, the Hamilton (2018)
regression filter, and Beveridge–Nelson decomposition for integrated series.

References:
- Hodrick, Prescott (1997); Ravn, Uhlig (2002) lambda = 6.25/1600 rule.
- Baxter, King (1999) approximate bandpass filter.
- Christiano, Fitzgerald (2003) random-walk optimal asymmetric filter.
- Hamilton (2018) "Why you should never use the HP filter" — h-step-ahead
    regression on lagged levels.
- Beveridge, Nelson (1981) permanent/transitory decomposition of I(1) series.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import linalg as spla

Array = NDArray[np.float64]


def _as_series(y: Array, min_len: int = 8) -> Array:
    v = np.asarray(y, dtype=float).reshape(-1)
    if v.size < min_len or not np.all(np.isfinite(v)):
        raise ValueError(f"series must be finite with length >= {min_len}")
    return v


def hp_filter(y: Array, lam: float = 1600.0) -> dict[str, Array]:
    """Hodrick–Prescott (1997) filter.

    Minimizes ``sum_t (y_t - tau_t)^2 + lam * sum_t ((tau_{t+1}-tau_t) -
    (tau_t - tau_{t-1}))^2`` — solved exactly via sparse second-difference
    matrix. Returns trend and cycle.
    """
    v = _as_series(y)
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be finite and positive")
    n = v.size
    d = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
    a = sparse.eye(n, format="csc") + lam * (d.T @ d)
    trend = spla.spsolve(a, v)
    return {"trend": trend, "cycle": v - trend}


def ravn_uhlig_lambda(periods_per_year: float) -> float:
    """Ravn–Uhlig (2002) lambda: ``1600 * (4 / periods_per_year)**4``.

    Quarterly 1600, annual 6.25, monthly 129600.
    """
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    return 1600.0 * (periods_per_year / 4.0) ** 4


def baxter_king(y: Array, low: float = 6.0, high: float = 32.0, k: int = 12) -> dict[str, Array]:
    """Baxter–King (1999) symmetric bandpass filter.

    ``low``/``high`` are cycle periods in bars (e.g. 6–32 quarters).
    Output is NaN in the first/last ``k`` entries (filter length burn-in).
    """
    v = _as_series(y, min_len=2)
    if not (np.isfinite(low) and np.isfinite(high) and 2.0 <= low < high):
        raise ValueError("need 2 <= low < high periods")
    if k < 1 or 2 * k + 1 >= v.size:
        raise ValueError("k too large for series length")
    j = np.arange(-k, k + 1, dtype=float)
    b = np.zeros(2 * k + 1)
    wl, wh = 2.0 * np.pi / high, 2.0 * np.pi / low
    for i, jj in enumerate(j):
        if jj == 0:
            b[i] = (wh - wl) / np.pi
        else:
            b[i] = (np.sin(wh * jj) - np.sin(wl * jj)) / (np.pi * jj)
    b[k] -= b.sum()  # zero-frequency adjustment: weights sum to 0
    cyc = np.full(v.size, np.nan)
    for t in range(k, v.size - k):
        cyc[t] = float(b @ v[t - k : t + k + 1])
    return {"cycle": cyc, "weights": b}


def christiano_fitzgerald(y: Array, low: float = 6.0, high: float = 32.0) -> dict[str, Array]:
    """Christiano–Fitzgerald (2003) asymmetric bandpass (random-walk version).

    Uses the full-sample asymmetric weights — valid to the endpoints (no
    burn-in NaNs). ``low``/``high`` are cycle periods in bars.
    """
    v = _as_series(y, min_len=4)
    if not (np.isfinite(low) and np.isfinite(high) and 2.0 <= low < high):
        raise ValueError("need 2 <= low < high periods")
    n = v.size
    wl, wh = 2.0 * np.pi / high, 2.0 * np.pi / low

    def bj(j: float) -> float:
        if j == 0:
            return (wh - wl) / np.pi
        return float((np.sin(wh * j) - np.sin(wl * j)) / (np.pi * j))

    cyc = np.zeros(n)
    for t in range(n):
        # Asymmetric CF weights over available leads/lags.
        j_min, j_max = -(t), (n - 1 - t)
        js = np.arange(j_min, j_max + 1, dtype=float)
        b = np.array([bj(j) for j in js])
        b[-j_min] -= b.sum()  # index of j=0; enforce weights summing to 0
        cyc[t] = float(b @ v[t + j_min : t + j_max + 1])
    return {"cycle": cyc}


def hamilton_filter(y: Array, h: int = 8, p: int = 4) -> dict[str, Array]:
    """Hamilton (2018) filter: regress y_t on (y_{t-h}, ..., y_{t-h-p+1}).

    Cycle = residual of the h-step-ahead autoregression; trend = fitted.
    Output has NaN warm-up for the first h+p-1 entries.
    """
    v = _as_series(y, min_len=4)
    if h < 1 or p < 1 or v.size <= h + p:
        raise ValueError("need h>=1, p>=1 and len > h+p")
    n = v.size
    rows, tgt = [], []
    for t in range(h + p - 1, n):
        rows.append([1.0] + [v[t - h - j] for j in range(p)])
        tgt.append(v[t])
    x = np.asarray(rows)
    tarr = np.asarray(tgt)
    beta, *_ = np.linalg.lstsq(x, tarr, rcond=None)
    fitted = x @ beta
    cyc = np.full(n, np.nan)
    trd = np.full(n, np.nan)
    cyc[h + p - 1 :] = tarr - fitted
    trd[h + p - 1 :] = fitted
    return {"cycle": cyc, "trend": trd, "beta": beta}


def beveridge_nelson(y: Array, ar_lags: int = 1) -> dict[str, Array]:
    """Beveridge–Nelson (1981) decomposition for an I(1) series.

    Fits AR(``ar_lags``) on Δy, then the BN permanent component is
    ``y_t + A(1)^{-1} * (fitted Δy_t)`` where A(1) = 1 - sum(phi).
    Cycle = -A(1)^{-1} * fitted Δy_t (the transitory component).
    """
    v = _as_series(y, min_len=6)
    if ar_lags < 1 or ar_lags > 8:
        raise ValueError("ar_lags must be in [1, 8]")
    dy = np.diff(v)
    n = dy.size
    if n <= ar_lags + 2:
        raise ValueError("series too short for ar_lags")
    x = np.column_stack(
        [np.ones(n - ar_lags)] + [dy[ar_lags - j - 1 : n - j - 1] for j in range(ar_lags)]
    )
    beta, *_ = np.linalg.lstsq(x, dy[ar_lags:], rcond=None)
    phi_sum = float(beta[1:].sum())
    if abs(1.0 - phi_sum) < 1e-6:
        raise ValueError("AR coefficients sum to ~1 — not BN-decomposable")
    fitted_dy = x @ beta
    adj = fitted_dy / (1.0 - phi_sum)
    permanent = np.full(v.size, np.nan)
    permanent[ar_lags + 1 :] = v[ar_lags + 1 :] + adj
    cycle = np.full(v.size, np.nan)
    cycle[ar_lags + 1 :] = -adj
    return {"permanent": permanent, "cycle": cycle, "phi_sum": np.array([phi_sum])}


def cf_random_walk_band(y: Array, low: float, high: float) -> Array:
    """Convenience: CF cycle component only (see ``christiano_fitzgerald``)."""
    return christiano_fitzgerald(y, low=low, high=high)["cycle"]


def corbae_ouliaris_band(y: Array, low: float = 6.0, high: float = 32.0) -> dict[str, Array]:
    """Corbae–Ouliaris (2006) frequency-domain bandpass (finite-sample DFT).

    Picks the DFT frequencies inside [2π/high, 2π/low], zeroes the rest,
    and inverts — no end-point loss, symmetric treatment.
    """
    v = _as_series(y, min_len=4)
    if not (np.isfinite(low) and np.isfinite(high) and 2.0 <= low < high):
        raise ValueError("need 2 <= low < high periods")
    n = v.size
    spec = np.fft.rfft(v - v.mean())
    freqs = np.fft.rfftfreq(n) * 2.0 * np.pi
    mask = (freqs >= 2.0 * np.pi / high) & (freqs <= 2.0 * np.pi / low)
    spec_f = np.where(mask, spec, 0.0)
    cyc = np.fft.irfft(spec_f, n=n)
    return {"cycle": cyc}
