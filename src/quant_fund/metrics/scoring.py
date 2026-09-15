"""Proper scoring rules and forecast metrics. See docs/MATH_SPEC.md."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def pinball_loss(y: Array, q: Array, tau: float) -> Array:
    """Elementwise pinball loss for quantile tau."""
    if not 0 < tau < 1:
        raise ValueError("tau must be in (0, 1)")
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    diff = y - q
    return np.maximum(tau * diff, (tau - 1.0) * diff)


def mean_pinball(y: Array, q: Array, tau: float) -> float:
    return float(np.mean(pinball_loss(y, q, tau)))


def coverage(y: Array, lower: Array, upper: Array) -> float:
    y = np.asarray(y, dtype=float)
    return float(np.mean((y >= lower) & (y <= upper)))


def interval_width(lower: Array, upper: Array) -> float:
    return float(np.mean(np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)))


def quantile_crossing_rate(quantiles: Array, taus: Array) -> float:
    """Fraction of rows with any Q_tau1 > Q_tau2 for tau1 < tau2.

    quantiles: (n, k) aligned with increasing taus.
    """
    q = np.asarray(quantiles, dtype=float)
    t = np.asarray(taus, dtype=float)
    if q.ndim != 2 or q.shape[1] != t.size:
        raise ValueError("quantiles must be (n, k) matching taus")
    diffs = np.diff(q, axis=1)
    crossed = np.any(diffs < -1e-15, axis=1)
    return float(np.mean(crossed))


def rearrange_quantiles(quantiles: Array) -> Array:
    """Sort along the quantile axis (rearrangement). Does not hide raw crossing rate."""
    return np.sort(np.asarray(quantiles, dtype=float), axis=1)


def crps_from_quantiles(y: Array, quantiles: Array, taus: Array) -> float:
    """Riemann-sum CRPS approximation from pinball losses (Gneiting-Raftery)."""
    q = np.asarray(quantiles, dtype=float)
    t = np.asarray(taus, dtype=float)
    y = np.asarray(y, dtype=float)
    if q.ndim != 2:
        raise ValueError("quantiles must be 2d")
    dt = np.diff(np.concatenate([[0.0], t]))
    total = np.zeros(y.shape[0], dtype=float)
    for k, tau in enumerate(t):
        total += pinball_loss(y, q[:, k], float(tau)) * dt[k]
    return float(np.mean(total))


def qlike(realized_var: Array, forecast_var: Array, floor: float = 1e-12) -> float:
    """QLIKE on variance: y/yhat - log(y/yhat) - 1."""
    y = np.asarray(realized_var, dtype=float)
    yhat = np.clip(np.asarray(forecast_var, dtype=float), floor, None)
    y = np.clip(y, floor, None)
    ratio = y / yhat
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def pearson_ic(pred: Array, realized: Array) -> float:
    p = np.asarray(pred, dtype=float)
    r = np.asarray(realized, dtype=float)
    mask = np.isfinite(p) & np.isfinite(r)
    if mask.sum() < 3:
        return float("nan")
    p, r = p[mask], r[mask]
    if np.std(p) == 0 or np.std(r) == 0:
        return float("nan")
    return float(np.corrcoef(p, r)[0, 1])


def rank_ic(pred: Array, realized: Array) -> float:
    p = np.asarray(pred, dtype=float)
    r = np.asarray(realized, dtype=float)
    mask = np.isfinite(p) & np.isfinite(r)
    if mask.sum() < 3:
        return float("nan")
    pr = _rankdata(p[mask])
    rr = _rankdata(r[mask])
    return pearson_ic(pr, rr)


def _rankdata(x: Array) -> Array:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, x.size + 1, dtype=float)
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.bincount(inv, weights=ranks)
        ranks = sums[inv] / counts[inv]
    return ranks


def icir(ics: Array, annualize: bool = False, periods_per_year: float = 252.0) -> float:
    x = np.asarray(ics, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2 or np.std(x, ddof=1) == 0:
        return float("nan")
    ir = float(np.mean(x) / np.std(x, ddof=1))
    if annualize:
        ir *= np.sqrt(periods_per_year)
    return ir


def pit_values(y: Array, quantiles: Array, taus: Array) -> Array:
    """Approximate PIT by interpolating the empirical CDF defined by quantiles."""
    q = np.asarray(quantiles, dtype=float)
    t = np.asarray(taus, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.empty(y.shape[0], dtype=float)
    for i in range(y.shape[0]):
        out[i] = float(np.interp(y[i], q[i], t, left=0.0, right=1.0))
    return out
