"""Coverage and width for conformal prediction sets. No Sharpe."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def conformal_quantile(scores: Array, alpha: float) -> float:
    """Finite-sample conformal quantile of residual scores.

    qhat = Quantile at level ceil((n+1)*(1-alpha))/n of s, clipped to 1.
    """
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 0.0
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    n = int(s.size)
    level = min(1.0, float(np.ceil((n + 1) * (1.0 - alpha)) / n))
    return float(np.quantile(s, level, method="higher"))


def cqr_scores(
    y: Array, lower: Array, upper: Array, scale: Array | None = None
) -> Array:
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    raw = np.maximum(lo - y, y - hi)
    if scale is None:
        return raw
    return raw / np.maximum(np.asarray(scale, dtype=float), 1e-12)


def onesided_scores(y: Array, bound: Array) -> Array:
    """Residual for an upper prediction bound: how far y exceeds the bound."""
    return np.asarray(y, dtype=float) - np.asarray(bound, dtype=float)


def expand_interval(
    lower: Array,
    upper: Array,
    qhat: float | Array,
    scale: Array | None = None,
) -> tuple[Array, Array]:
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    q = np.asarray(qhat, dtype=float)
    if scale is None:
        return lo - q, hi + q
    sc = np.maximum(np.asarray(scale, dtype=float), 1e-12)
    return lo - q * sc, hi + q * sc


def covered(y: Array, lower: Array, upper: Array) -> Array:
    y = np.asarray(y, dtype=float)
    return ((y >= np.asarray(lower, dtype=float)) & (y <= np.asarray(upper, dtype=float))).astype(
        float
    )


@dataclass(frozen=True)
class SetMetrics:
    coverage: float
    mean_width: float
    median_width: float
    n: int


def set_metrics(y: Array, lower: Array, upper: Array) -> SetMetrics:
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if int(mask.sum()) == 0:
        return SetMetrics(float("nan"), float("nan"), float("nan"), 0)
    y, lo, hi = y[mask], lo[mask], hi[mask]
    width = hi - lo
    return SetMetrics(
        coverage=float(np.mean((y >= lo) & (y <= hi))),
        mean_width=float(np.mean(width)),
        median_width=float(np.median(width)),
        n=int(y.size),
    )


def conditional_coverage(
    y: Array, lower: Array, upper: Array, labels: Array
) -> dict[str, float]:
    """Coverage within each label. Labels are never mixed across groups."""
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    lab = np.asarray(labels)
    out: dict[str, float] = {}
    for g in np.unique(lab):
        sel = lab == g
        if int(sel.sum()) < 5:
            continue
        out[str(g)] = float(np.mean((y[sel] >= lo[sel]) & (y[sel] <= hi[sel])))
    return out


def worst_slice_coverage(cond: dict[str, float]) -> float:
    if not cond:
        return float("nan")
    return float(min(cond.values()))


def assign_terciles(
    values: Array, cuts: Array | None = None, prefix: str = "vol"
) -> tuple[np.ndarray, Array]:
    """PIT-safe tercile labels. Cuts must come from train/cal, never test y."""
    v = np.asarray(values, dtype=float)
    finite = v[np.isfinite(v)]
    if cuts is None:
        mid = float(np.nanmedian(finite)) if finite.size else 0.0
        if finite.size < 6:
            cuts = np.array([mid, mid])
        else:
            cuts = np.nanquantile(finite, [1.0 / 3.0, 2.0 / 3.0])
    cuts = np.asarray(cuts, dtype=float)
    out = np.full(v.size, f"mid_{prefix}", dtype=object)
    out[v <= cuts[0]] = f"low_{prefix}"
    out[v > cuts[1]] = f"high_{prefix}"
    return out, cuts
