"""Bai-Perron multiple structural-break detection in regression.

Exact optimal m-partitions by one-pass dynamic programming (Bai & Perron
1998, section 4) and sequential ``sup-Wald`` break tests against the
Bai-Perron (2003) critical values.  The design is the classical one:
``y_t = x_t' beta_j + e_t`` with coefficients free to shift at each break;
segments must be at least ``trim * T`` observations.

References: J. Bai and P. Perron (1998), Econometrica 66(1); J. Bai and
P. Perron (2003), Journal of Applied Econometrics 18(1).  Fail-closed on
non-finite input, too few observations, or degenerate segment regressions.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

# Bai & Perron (2003), Table 1, sup-Wald critical values at eps = 0.15
# trimming, rows = (10%, 5%, 1%), columns q = 1..3 regressors.
_SUPF_CV = {
    0.10: {1: 7.12, 2: 10.18, 3: 11.13},
    0.05: {1: 8.85, 2: 12.16, 3: 13.25},
    0.01: {1: 12.16, 2: 14.61, 3: 15.73},
}


def _as_xy(y: Array, x: Array) -> tuple[Array, Array]:
    ya = np.asarray(y, dtype=float).ravel()
    xa = np.asarray(x, dtype=float)
    if xa.ndim == 1:
        xa = xa.reshape(-1, 1)
    t, k = xa.shape
    if ya.shape[0] != t:
        raise ValueError("y and x must share the same row count")
    if t < 8:
        raise ValueError("need at least 8 observations for break testing")
    if k < 1 or not np.all(np.isfinite(ya)) or not np.all(np.isfinite(xa)):
        raise ValueError("y/x must be finite with at least one regressor")
    return ya, xa


def _seg_ssr(y: Array, x: Array) -> float:
    """OLS residual sum of squares for one segment."""
    n = y.shape[0]
    if n <= x.shape[1]:
        return float("inf")
    b, *_ = np.linalg.lstsq(x, y, rcond=None)
    r = y - x @ b
    return float(r @ r)


def _ssr_matrix(y: Array, x: Array, h: int) -> Array:
    """``ssr[i, j]`` = SSR of OLS fit on rows ``i..j-1`` (>= h long)."""
    t = y.shape[0]
    ssr = np.full((t + 1, t + 1), np.inf)
    for i in range(0, t - h + 1):
        for j in range(i + h, t + 1):
            ssr[i, j] = _seg_ssr(y[i:j], x[i:j])
    return ssr


def sup_wald(y: Array, x: Array, *, trim: float = 0.15) -> dict[str, float]:
    """Supremum Wald statistic for a single break in ``y ~ x``.

    Scans candidate breaks in ``[trim*T, T - trim*T]`` and reports the max
    F-type statistic plus the argmax break index.  ``q = x.shape[1]``.
    """
    ya, xa = _as_xy(y, x)
    t, q = xa.shape
    if not 0.0 < trim < 0.5:
        raise ValueError("trim must be in (0, 0.5)")
    lo = max(1, int(math.floor(t * trim)))
    hi = t - lo
    if hi <= lo:
        raise ValueError("trim leaves no candidate breaks")
    full = _seg_ssr(ya, xa)
    best_stat, best_b = -np.inf, -1
    for b in range(lo, hi + 1):
        s1 = _seg_ssr(ya[:b], xa[:b])
        s2 = _seg_ssr(ya[b:], xa[b:])
        den = (s1 + s2) / max(t - 2 * q, 1)
        if den <= 0 or not np.isfinite(den):
            continue
        stat = (full - s1 - s2) / q / den
        if stat > best_stat:
            best_stat, best_b = stat, b
    return {"stat": float(best_stat), "break": float(best_b), "q": float(q)}


def sup_wald_cv(q: int, alpha: float = 0.05) -> float:
    """Bai-Perron (2003) 5% critical value for sup-Wald at eps=0.15."""
    if alpha not in _SUPF_CV:
        raise ValueError("alpha must be one of 0.10, 0.05, 0.01")
    qq = min(max(int(q), 1), 3)
    return _SUPF_CV[alpha][qq]


def breakpoints_dp(y: Array, x: Array, m: int, *, trim: float = 0.15) -> dict[str, Array]:
    """Exact optimal ``m``-break partition via Bai-Perron dynamic programming.

    Returns break indices ``(b1 < ... < bm)``, per-segment SSR, and total SSR.
    ``m = 0`` degenerates to the full-sample regression.
    """
    ya, xa = _as_xy(y, x)
    t, k = xa.shape
    if not 0.0 < trim < 0.5:
        raise ValueError("trim must be in (0, 0.5)")
    h = max(k + 1, int(math.floor(t * trim)))
    if m < 0 or m > 20:
        raise ValueError("m must be in [0, 20]")
    if m * h > t - h:
        raise ValueError(f"m={m} breaks infeasible with trim={trim}")
    if m == 0:
        return {
            "breaks": np.zeros(0, dtype=int),
            "ssr_total": np.array([_seg_ssr(ya, xa)]),
            "ssr_seg": np.array([_seg_ssr(ya, xa)]),
        }
    ssr = _ssr_matrix(ya, xa, h)
    # dp[s, j] = minimal SSR covering rows 0..j-1 with s breaks (s segments
    # fully inside, the last ends at j). parent[s, j] = break index i.
    dp = np.full((m + 1, t + 1), np.inf)
    parent = np.full((m + 1, t + 1), -1, dtype=int)
    dp[0] = ssr[0]
    for s in range(1, m + 1):
        for j in range((s + 1) * h, t + 1):
            lo = s * h
            hi = j - h
            cand = dp[s - 1, lo:hi] + ssr[lo:hi, j]
            if not np.isfinite(cand).any():
                continue
            i = int(np.argmin(cand)) + lo
            dp[s, j] = float(cand[i - lo])
            parent[s, j] = i
    if not np.isfinite(dp[m, t]):
        raise ValueError("no feasible partition found")
    breaks: list[int] = []
    j = t
    for s in range(m, 0, -1):
        i = parent[s, j]
        if i < 0:
            raise ValueError("no feasible partition found")
        breaks.append(i)
        j = i
    breaks.reverse()
    bounds = [0, *breaks, t]
    seg_ssr = np.array(
        [_seg_ssr(ya[a:b], xa[a:b]) for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
    )
    return {
        "breaks": np.array(breaks, dtype=int),
        "ssr_total": np.array([float(seg_ssr.sum())]),
        "ssr_seg": seg_ssr,
    }


def sequential_breaks(
    y: Array,
    x: Array,
    *,
    m_max: int = 5,
    trim: float = 0.15,
    alpha: float = 0.05,
) -> dict[str, Array]:
    """Sequential ``l -> l+1`` break detection (Bai-Perron sequential method).

    At each step the segment contributing the largest sup-Wald statistic is
    split if it exceeds the ``alpha`` critical value; then the full set of
    breakpoints is re-optimized by DP.
    """
    ya, xa = _as_xy(y, x)
    t, q = xa.shape
    cv = sup_wald_cv(q, alpha)
    breaks: list[int] = []
    for _m in range(m_max):
        bounds = [0, *sorted(breaks), t]
        cand_stat, cand_b = -np.inf, -1
        for a, b in zip(bounds[:-1], bounds[1:], strict=True):
            if b - a < 2 * max(1, int(math.floor(t * trim))):
                continue
            res = sup_wald(ya[a:b], xa[a:b], trim=trim)
            if res["stat"] > cand_stat:
                cand_stat, cand_b = res["stat"], a + int(res["break"])
        if cand_stat <= cv or cand_b < 0:
            break
        breaks.append(cand_b)
        opt = breakpoints_dp(ya, xa, len(breaks), trim=trim)
        breaks = sorted(int(v) for v in opt["breaks"])
    opt = breakpoints_dp(ya, xa, len(breaks), trim=trim)
    return {
        "breaks": opt["breaks"],
        "n_breaks": np.array([len(breaks)]),
        "ssr_total": opt["ssr_total"],
        "cv": np.array([cv]),
        "last_sup_wald": np.array([cand_stat]),
    }


def refit_segments(y: Array, x: Array, breaks: Array) -> dict[str, Array]:
    """Per-segment OLS coefficients and SSR given a breakpoint vector."""
    ya, xa = _as_xy(y, x)
    b_idx = np.asarray(breaks, dtype=int).ravel()
    t = ya.shape[0]
    if (b_idx < 0).any() or (b_idx >= t).any() or np.any(np.diff(b_idx) <= 0):
        raise ValueError("breaks must be strictly increasing interior indices")
    bounds = [0, *b_idx.tolist(), t]
    coefs: list[Array] = []
    ssrs: list[float] = []
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        coef, *_ = np.linalg.lstsq(xa[a:b], ya[a:b], rcond=None)
        coefs.append(coef)
        ssrs.append(_seg_ssr(ya[a:b], xa[a:b]))
    coef_mat = np.stack(coefs)
    total = float(np.sum(ssrs))
    k = xa.shape[1] * (len(bounds) - 1)
    bic = float(t * math.log(max(total / t, 1e-12)) + k * math.log(t))
    return {
        "coefs": coef_mat,
        "ssr_seg": np.array(ssrs),
        "ssr_total": np.array([total]),
        "bic": np.array([bic]),
        "n_segments": np.array([len(bounds) - 1]),
    }
