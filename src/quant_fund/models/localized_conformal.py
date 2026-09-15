"""Localized kernel-weighted split CQR (Lei–Wasserman / Guan).

RBF weights on a PIT-safe 1-d covariate (vol). Smoother than Mondrian
bins (ADR-009) and complementary to likelihood-ratio weighted CQR (ADR-013).
Lab scores are coverage and width. No Sharpe.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.conformal import (
    conformal_quantile,
    cqr_scores,
    expand_interval,
    set_metrics,
)
from quant_fund.models.base import JoblibMixin, ModelMeta

Array = NDArray[np.float64]

WEIGHT_CLIP: tuple[float, float] = (1e-3, 1e3)
MIN_ESS: float = 12.0
_BANDWIDTH_SUBSAMPLE: int = 400


def _as_1d(values: Array) -> Array:
    return np.asarray(values, dtype=float).ravel()


def calibration_bandwidth(x_cal: Array) -> float:
    """Median positive pairwise |x_i - x_j| on calibration vol."""
    x = _as_1d(x_cal)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 1.0
    xs = np.sort(x)
    if xs.size > _BANDWIDTH_SUBSAMPLE:
        idx = np.linspace(0, xs.size - 1, _BANDWIDTH_SUBSAMPLE).astype(int)
        xs = xs[idx]
    diffs = np.abs(xs[:, None] - xs[None, :])
    off = diffs[np.triu_indices(int(xs.size), k=1)]
    pos = off[off > 0]
    if pos.size == 0:
        return 1.0
    return float(max(np.median(pos), 1e-6))


def rbf_weights(x_cal: Array, x: float, bandwidth: float) -> Array:
    """w_i(x) = exp(-((x_i - x) / h)^2). Non-finite cal rows get weight 0."""
    h = float(bandwidth)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("bandwidth must be > 0")
    xc = _as_1d(x_cal)
    u = (xc - float(x)) / h
    w = np.exp(-np.square(u))
    return np.where(np.isfinite(xc), w, 0.0)


def clip_weights(weights: Array, clip: tuple[float, float] = WEIGHT_CLIP) -> Array:
    w = _as_1d(weights)
    lo, hi = float(clip[0]), float(clip[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi < lo or lo < 0.0:
        raise ValueError("weight clip must be a finite 0 <= lo <= hi range")
    out = np.where(np.isfinite(w), w, 0.0)
    return np.clip(out, lo, hi)


def effective_sample_size(weights: Array) -> float:
    """Kish n_eff = (sum w)^2 / sum(w^2) on positive finite weights."""
    w = _as_1d(weights)
    w = w[np.isfinite(w) & (w > 0.0)]
    if w.size == 0:
        return 0.0
    s1 = float(np.sum(w))
    s2 = float(np.sum(w * w))
    if s2 <= 0.0:
        return 0.0
    return (s1 * s1) / s2


def _weighted_quantile_higher(values: Array, weights: Array, level: float) -> float:
    """Weighted analog of ``np.quantile(..., method='higher')``."""
    order = np.argsort(values, kind="mergesort")
    v = values[order]
    w = weights[order]
    n = int(v.size)
    if n == 1:
        return float(v[0])
    q = float(np.clip(level, 0.0, 1.0))
    total = float(np.sum(w))
    starts = np.zeros(n, dtype=float)
    starts[1:] = (n * np.cumsum(w)[:-1]) / total
    idx = int(np.searchsorted(starts, q * (n - 1), side="left"))
    return float(v[min(idx, n - 1)])


def localized_conformal_quantile(scores: Array, weights: Array, alpha: float) -> float:
    """Finite-sample weighted conformal quantile of residual scores.

    Same (1 + sum w) level as Tibshirani weighted conformal: rescale w to
    mean 1 and take the ``higher`` weighted quantile at

        min(1, ceil((1-alpha) * (1 + sum w)) / sum w).

    Uniform weights match ``conformal_quantile``.
    """
    s = _as_1d(scores)
    w = _as_1d(weights)
    if s.size != w.size:
        raise ValueError("scores and weights must have the same length")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    mask = np.isfinite(s) & np.isfinite(w) & (w > 0.0)
    s, w = s[mask], w[mask]
    if s.size == 0:
        return 0.0
    n = int(s.size)
    w = w * (n / float(np.sum(w)))
    total = float(np.sum(w))
    level = min(1.0, float(np.ceil((1.0 - alpha) * (1.0 + total)) / total))
    return _weighted_quantile_higher(s, w, level)


def _synthetic_het_vol(
    n_cal: int, n_test: int, seed: int
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    """Exchangeable two-component vol mixture (low 0.4, high 2.0)."""
    rng = np.random.default_rng(seed)
    n = int(n_cal) + int(n_test)
    vol_low, vol_high, band = 0.4, 2.0, 0.25
    x = rng.choice(np.array([vol_low, vol_high]), size=n)
    y = rng.normal(0.0, x)
    lo = np.full(n, -band)
    hi = np.full(n, band)
    return (
        y[:n_cal],
        lo[:n_cal],
        hi[:n_cal],
        x[:n_cal],
        y[n_cal:],
        lo[n_cal:],
        hi[n_cal:],
        x[n_cal:],
    )


class LocalizedCQR(JoblibMixin):
    """Split CQR with Lei–Wasserman / Guan RBF-weighted residual quantiles."""

    def __init__(
        self,
        alpha: float = 0.10,
        bandwidth: float | None = None,
        min_ess: float = MIN_ESS,
        weight_clip: tuple[float, float] = WEIGHT_CLIP,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if bandwidth is not None and (not np.isfinite(bandwidth) or float(bandwidth) <= 0.0):
            raise ValueError("bandwidth must be > 0")
        if float(min_ess) <= 0.0:
            raise ValueError("min_ess must be > 0")
        self.alpha = float(alpha)
        self.bandwidth = None if bandwidth is None else float(bandwidth)
        self.min_ess = float(min_ess)
        self.weight_clip = (float(weight_clip[0]), float(weight_clip[1]))
        self.bandwidth_: float = 1.0
        self.global_qhat = 0.0
        self.qhat: float | Array = 0.0
        self.scores_: Array | None = None
        self.x_cal_: Array | None = None

    def calibrate(self, y: Array, lo: Array, hi: Array, x_cal: Array) -> LocalizedCQR:
        s = cqr_scores(y, lo, hi)
        x = _as_1d(x_cal)
        if x.size != s.size:
            raise ValueError("x_cal must align with y")
        self.scores_ = s
        self.x_cal_ = x
        self.global_qhat = conformal_quantile(s, self.alpha)
        self.bandwidth_ = self.bandwidth if self.bandwidth is not None else calibration_bandwidth(x)
        return self

    def _qhat_at(self, x: float) -> float:
        if self.scores_ is None or self.x_cal_ is None:
            return 0.0
        if not np.isfinite(x):
            return float(self.global_qhat)
        w = clip_weights(rbf_weights(self.x_cal_, x, self.bandwidth_), self.weight_clip)
        if effective_sample_size(w) < self.min_ess:
            return float(self.global_qhat)
        return localized_conformal_quantile(self.scores_, w, self.alpha)

    def _qhat_rows(self, x_query: Array) -> Array:
        x_q = _as_1d(x_query)
        n = int(x_q.size)
        if self.scores_ is None or self.x_cal_ is None or n == 0:
            return np.zeros(n, dtype=float)
        uniq, inv = np.unique(x_q, return_inverse=True)
        local = np.asarray([self._qhat_at(float(xv)) for xv in uniq], dtype=float)
        return local[inv]

    def predict_sets(self, lo: Array, hi: Array, x_query: Array) -> tuple[Array, Array]:
        lower = np.asarray(lo, dtype=float)
        upper = np.asarray(hi, dtype=float)
        x_q = _as_1d(x_query)
        if x_q.size == 1 and lower.size > 1:
            x_q = np.full(lower.size, x_q[0])
        if x_q.size != lower.size:
            raise ValueError("x_query must align with lo/hi")
        qhats = self._qhat_rows(x_q)
        self.qhat = qhats
        return expand_interval(lower, upper, qhats)

    def metadata(self) -> ModelMeta:
        extra: dict[str, Any] = {
            "alpha": self.alpha,
            "bandwidth": self.bandwidth_,
            "min_ess": self.min_ess,
            "weight_clip": list(self.weight_clip),
            "global_qhat": self.global_qhat,
        }
        return ModelMeta(
            family="conformal",
            name="localized_cqr",
            version="v1",
            extra=extra,
        )


def bench_localized_cqr(
    n_cal: int = 600,
    n_test: int = 400,
    alpha: float = 0.10,
    seed: int = 21,
) -> dict[str, float]:
    """Coverage and high/low-vol widths on an exchangeable vol mixture. No Sharpe."""
    y_c, lo_c, hi_c, x_c, y_t, lo_t, hi_t, x_t = _synthetic_het_vol(n_cal, n_test, seed)
    model = LocalizedCQR(alpha).calibrate(y_c, lo_c, hi_c, x_c)
    plo, phi = model.predict_sets(lo_t, hi_t, x_t)
    local = set_metrics(y_t, plo, phi)
    width = phi - plo
    high = x_t > 1.0
    low = ~high
    q_g = conformal_quantile(cqr_scores(y_c, lo_c, hi_c), alpha)
    glo, ghi = expand_interval(lo_t, hi_t, q_g)
    plain = set_metrics(y_t, glo, ghi)
    return {
        "coverage": local.coverage,
        "mean_width": local.mean_width,
        "median_width": local.median_width,
        "high_vol_mean_width": float(np.mean(width[high])) if bool(high.any()) else float("nan"),
        "low_vol_mean_width": float(np.mean(width[low])) if bool(low.any()) else float("nan"),
        "global_mean_width": plain.mean_width,
        "global_coverage": plain.coverage,
        "n": float(local.n),
        "alpha": float(alpha),
        "bandwidth": float(model.bandwidth_),
        "seed": float(seed),
    }
