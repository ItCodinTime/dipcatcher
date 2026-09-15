"""Proper scores for probabilities, PIT, and VaR hit tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Array = NDArray[np.float64]


def brier_score(prob: Array, y: Array) -> float:
    p = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
    t = np.asarray(y, dtype=float)
    mask = np.isfinite(p) & np.isfinite(t)
    if int(mask.sum()) < 1:
        return float("nan")
    return float(np.mean((p[mask] - t[mask]) ** 2))


def log_loss(prob: Array, y: Array, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(prob, dtype=float), eps, 1.0 - eps)
    t = np.asarray(y, dtype=float)
    mask = np.isfinite(p) & np.isfinite(t)
    if int(mask.sum()) < 1:
        return float("nan")
    yy = np.clip(t[mask], 0.0, 1.0)
    return float(-np.mean(yy * np.log(p[mask]) + (1.0 - yy) * np.log(1.0 - p[mask])))


def expected_calibration_error(prob: Array, y: Array, n_bins: int = 10) -> float:
    p = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
    t = np.asarray(y, dtype=float)
    mask = np.isfinite(p) & np.isfinite(t)
    p, t = p[mask], t[mask]
    if p.size < n_bins:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        sel = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
        if not sel.any():
            continue
        ece += float(sel.mean()) * abs(float(t[sel].mean()) - float(p[sel].mean()))
    return float(ece)


def kupiec_pof(hits: Array, alpha: float) -> tuple[float, float, float]:
    """Kupiec proportion-of-failures test. Returns (hit_rate, lr, p_value)."""
    h = np.asarray(hits, dtype=float)
    h = h[np.isfinite(h)]
    n = int(h.size)
    x = int(np.sum(h > 0.5))
    if n < 10 or not 0 < alpha < 1:
        return float("nan"), float("nan"), float("nan")
    rate = x / n
    if x == 0 or x == n:
        return rate, float("nan"), float("nan")
    lr = -2.0 * (
        (n - x) * np.log(1.0 - alpha)
        + x * np.log(alpha)
        - (n - x) * np.log(1.0 - rate)
        - x * np.log(rate)
    )
    p = float(stats.chi2.sf(lr, df=1))
    return float(rate), float(lr), p


def pit_ks(pits: Array) -> tuple[float, float]:
    """Kolmogorov–Smirnov test of PIT against Uniform(0, 1)."""
    u = np.asarray(pits, dtype=float)
    u = u[np.isfinite(u)]
    u = np.clip(u, 1e-9, 1.0 - 1e-9)
    if u.size < 8:
        return float("nan"), float("nan")
    stat, p = stats.kstest(u, "uniform")
    return float(stat), float(p)
