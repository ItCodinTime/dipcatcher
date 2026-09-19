"""Weighted split CQR under covariate shift (Tibshirani et al. 2019).

Likelihood-ratio weights w(x) = dP_test / dP_cal on a PIT-safe 1-d
covariate (vol). Lab scores are coverage and width. No Sharpe.
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


def _as_1d(x: Array) -> Array:
    return np.asarray(x, dtype=float).ravel()


def _histogram_edges(x_cal: Array, x_test: Array, bins: int) -> Array | None:
    finite = np.concatenate([x_cal[np.isfinite(x_cal)], x_test[np.isfinite(x_test)]])
    if finite.size == 0:
        return None
    lo = float(finite.min())
    hi = float(finite.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    pad = 1e-12 * max(abs(hi), abs(lo), 1.0)
    return np.linspace(lo - pad, hi + pad, int(bins) + 1)


def _bin_index(x: Array, edges: Array) -> Array:
    idx = np.digitize(x, edges, right=False) - 1
    return np.clip(idx, 0, int(edges.size) - 2)


def likelihood_ratio_weights(x_cal: Array, x_test_or_query: Array, bins: int = 8) -> Array:
    """Histogram density ratio p_test / p_cal evaluated on calibration rows.

    Laplace-smoothed 1-d histograms on a shared grid. Weights are clipped to
    ``WEIGHT_CLIP`` so they stay positive and finite.
    """
    if int(bins) < 1:
        raise ValueError("bins must be >= 1")
    x_c = _as_1d(x_cal)
    x_t = _as_1d(x_test_or_query)
    ones = np.ones(x_c.size, dtype=float)
    edges = _histogram_edges(x_c, x_t, int(bins))
    if edges is None:
        return np.clip(ones, WEIGHT_CLIP[0], WEIGHT_CLIP[1])
    finite_c = x_c[np.isfinite(x_c)]
    finite_t = x_t[np.isfinite(x_t)]
    if finite_c.size == 0 or finite_t.size == 0:
        return np.clip(ones, WEIGHT_CLIP[0], WEIGHT_CLIP[1])
    c_counts, _ = np.histogram(finite_c, bins=edges)
    t_counts, _ = np.histogram(finite_t, bins=edges)
    c_p = (c_counts + 1.0) / float((c_counts + 1.0).sum())
    t_p = (t_counts + 1.0) / float((t_counts + 1.0).sum())
    ratio = t_p / np.maximum(c_p, 1e-12)
    out = ratio[_bin_index(x_c, edges)]
    out = np.where(np.isfinite(x_c), out, 1.0)
    return np.clip(out, WEIGHT_CLIP[0], WEIGHT_CLIP[1])


def weighted_conformal_quantile(scores: Array, weights: Array, alpha: float) -> float:
    """Finite-sample weighted conformal quantile of residual scores.

    Tibshirani et al. (2019): sort by score and take the first ``s_i`` whose
    normalized cumulative weight ``sum_{j<=i} w_j / (sum_j w_j + 1)`` reaches
    ``1 - alpha`` (the leftover ``1 / (sum_j w_j + 1)`` is point mass at
    ``+inf``). Uniform weights reproduce the finite-sample split-conformal
    order statistic of ``conformal_quantile``. When the level is not
    attainable from the calibration sample we clip to the sample max so
    downstream intervals stay finite (this cannot cover at ``1 - alpha``).
    """
    s = _as_1d(scores)
    w = _as_1d(weights)
    if s.size != w.size:
        raise ValueError("scores and weights must have the same length")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    mask = np.isfinite(s) & np.isfinite(w) & (w > 0)
    s, w = s[mask], w[mask]
    if s.size == 0:
        return 0.0
    n = int(s.size)
    w = w * (n / float(np.sum(w)))
    order = np.argsort(s, kind="mergesort")
    s_sorted, w_sorted = s[order], w[order]
    cumulative = np.cumsum(w_sorted) / (float(np.sum(w_sorted)) + 1.0)
    idx = int(np.searchsorted(cumulative, 1.0 - alpha, side="left"))
    if idx >= n:
        return float(s_sorted[-1])
    return float(s_sorted[idx])


def _synthetic_vol_shift(
    n_cal: int, n_test: int, seed: int
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    """Two-component vol mixture: low-vol-heavy cal, high-vol-heavy test."""
    rng = np.random.default_rng(seed)
    vol_low, vol_high, band = 0.4, 2.0, 0.25
    x_cal = np.where(rng.random(n_cal) < 0.15, vol_high, vol_low)
    x_te = np.where(rng.random(n_test) < 0.85, vol_high, vol_low)
    y_cal = rng.normal(0.0, x_cal)
    y_te = rng.normal(0.0, x_te)
    lo_c = np.full(n_cal, -band)
    hi_c = np.full(n_cal, band)
    lo_t = np.full(n_test, -band)
    hi_t = np.full(n_test, band)
    return y_cal, lo_c, hi_c, x_cal, y_te, lo_t, hi_t, x_te


class WeightedSplitCQR(JoblibMixin):
    """Split CQR with Tibshirani–Barber–Candès–Ramdas likelihood-ratio weights."""

    def __init__(self, alpha: float = 0.10, bins: int = 8) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if int(bins) < 1:
            raise ValueError("bins must be >= 1")
        self.alpha = float(alpha)
        self.bins = int(bins)
        self.qhat: float | Array = 0.0
        self.scores_: Array | None = None
        self.x_cal_: Array | None = None

    def calibrate(self, y: Array, lower: Array, upper: Array, x_cal: Array) -> WeightedSplitCQR:
        s = cqr_scores(y, lower, upper)
        x = _as_1d(x_cal)
        if x.size != s.size:
            raise ValueError("x_cal must align with y")
        self.scores_ = s
        self.x_cal_ = x
        return self

    def _qhat_rows(self, x_query: Array) -> Array:
        x_q = _as_1d(x_query)
        n = int(x_q.size)
        if self.scores_ is None or self.x_cal_ is None or n == 0:
            return np.zeros(n, dtype=float)
        edges = _histogram_edges(self.x_cal_, x_q, self.bins)
        qhats = np.zeros(n, dtype=float)
        if edges is None:
            w = np.ones(self.x_cal_.size, dtype=float)
            qhats[:] = weighted_conformal_quantile(self.scores_, w, self.alpha)
            return qhats
        bins = _bin_index(x_q, edges)
        for b in np.unique(bins):
            sel = bins == b
            w = likelihood_ratio_weights(self.x_cal_, x_q[sel], bins=self.bins)
            qhats[sel] = weighted_conformal_quantile(self.scores_, w, self.alpha)
        return qhats

    def predict_sets(self, lower: Array, upper: Array, x_query: Array) -> tuple[Array, Array]:
        lo = np.asarray(lower, dtype=float)
        hi = np.asarray(upper, dtype=float)
        x_q = _as_1d(x_query)
        if x_q.size == 1 and lo.size > 1:
            x_q = np.full(lo.size, x_q[0])
        if x_q.size != lo.size:
            raise ValueError("x_query must align with lower/upper")
        qhats = self._qhat_rows(x_q)
        self.qhat = qhats
        return expand_interval(lo, hi, qhats)

    def metadata(self) -> ModelMeta:
        extra: dict[str, Any] = {"alpha": self.alpha, "bins": self.bins}
        return ModelMeta(
            family="conformal",
            name="weighted_split_cqr",
            version="v1",
            extra=extra,
        )


def bench_weighted_cqr(
    n_cal: int = 800,
    n_test: int = 400,
    alpha: float = 0.10,
    seed: int = 7,
) -> dict[str, float]:
    """Coverage and width on a planted two-component vol shift. No Sharpe."""
    y_cal, lo_c, hi_c, x_cal, y_te, lo_t, hi_t, x_te = _synthetic_vol_shift(n_cal, n_test, seed)
    model = WeightedSplitCQR(alpha).calibrate(y_cal, lo_c, hi_c, x_cal)
    wlo, whi = model.predict_sets(lo_t, hi_t, x_te)
    q_u = conformal_quantile(cqr_scores(y_cal, lo_c, hi_c), alpha)
    ulo, uhi = expand_interval(lo_t, hi_t, q_u)
    weighted = set_metrics(y_te, wlo, whi)
    plain = set_metrics(y_te, ulo, uhi)
    return {
        "coverage": weighted.coverage,
        "mean_width": weighted.mean_width,
        "median_width": weighted.median_width,
        "unweighted_coverage": plain.coverage,
        "unweighted_mean_width": plain.mean_width,
        "unweighted_median_width": plain.median_width,
        "n": float(weighted.n),
        "alpha": float(alpha),
    }
