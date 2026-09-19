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


def localized_conformal_quantile(scores: Array, weights: Array, alpha: float) -> float:
    """Finite-sample weighted conformal quantile of residual scores.

    Tibshirani et al. (2019) normalized-cumulative-weight quantile; uniform
    weights reproduce the finite-sample split-conformal order statistic of
    ``conformal_quantile``. When the level is not attainable from the
    calibration sample we clip to the sample max so downstream intervals
    stay finite (this cannot cover at ``1 - alpha``).
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
    order = np.argsort(s, kind="mergesort")
    s_sorted, w_sorted = s[order], w[order]
    cumulative = np.cumsum(w_sorted) / (float(np.sum(w_sorted)) + 1.0)
    idx = int(np.searchsorted(cumulative, 1.0 - alpha, side="left"))
    if idx >= n:
        return float(s_sorted[-1])
    return float(s_sorted[idx])


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
    *,
    y_cal: Array | None = None,
    lo_cal: Array | None = None,
    hi_cal: Array | None = None,
    x_cal: Array | None = None,
    y_test: Array | None = None,
    lo_test: Array | None = None,
    hi_test: Array | None = None,
    x_test: Array | None = None,
    dates_test: Array | None = None,
    dgp: str | None = None,
) -> dict[str, float | str]:
    """Localized CQR coverage/widths. Panel arrays preferred; toy path is ``dgp=fixture``."""
    panel_args = (y_cal, lo_cal, hi_cal, x_cal, y_test, lo_test, hi_test, x_test)
    if any(a is not None for a in panel_args):
        if any(a is None for a in panel_args):
            raise ValueError("pass all panel cal/test arrays or none")
        y_c, lo_c, hi_c, x_c = y_cal, lo_cal, hi_cal, x_cal  # type: ignore[assignment]
        y_t, lo_t, hi_t, x_t = y_test, lo_test, hi_test, x_test  # type: ignore[assignment]
        dgp_label = dgp or "panel"
    else:
        y_c, lo_c, hi_c, x_c, y_t, lo_t, hi_t, x_t = _synthetic_het_vol(n_cal, n_test, seed)
        dgp_label = "fixture"
    y_c = np.asarray(y_c, dtype=float)
    lo_c = np.asarray(lo_c, dtype=float)
    hi_c = np.asarray(hi_c, dtype=float)
    x_c = np.asarray(x_c, dtype=float)
    y_t = np.asarray(y_t, dtype=float)
    lo_t = np.asarray(lo_t, dtype=float)
    hi_t = np.asarray(hi_t, dtype=float)
    x_t = np.asarray(x_t, dtype=float)
    model = LocalizedCQR(alpha).calibrate(y_c, lo_c, hi_c, x_c)
    plo, phi = model.predict_sets(lo_t, hi_t, x_t)
    local = set_metrics(y_t, plo, phi)
    width = phi - plo
    if dgp_label == "fixture":
        # Toy mixture is coded around the 1.0 vol threshold (see _synthetic_het_vol).
        high = np.asarray(x_t, dtype=float) > 1.0
        low = ~high
    else:
        med_x = float(np.nanmedian(x_t)) if np.isfinite(x_t).any() else 1.0
        # Keep the upper atom when the median equals the high regime.
        high = np.isfinite(x_t) & (x_t >= med_x)
        low = np.isfinite(x_t) & (x_t < med_x)
    q_g = conformal_quantile(cqr_scores(y_c, lo_c, hi_c), alpha)
    glo, ghi = expand_interval(lo_t, hi_t, q_g)
    plain = set_metrics(y_t, glo, ghi)
    from quant_fund.metrics.conformal import covered
    from quant_fund.metrics.probability import kupiec_pof

    hits = 1.0 - covered(y_t, plo, phi)
    hits = hits[np.isfinite(hits)]
    if hits.size >= 10:
        rate, lr, kp = kupiec_pof(hits, alpha)
    else:
        rate, lr, kp = float("nan"), float("nan"), float("nan")
    out: dict[str, float | str] = {
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
        "miss_rate": rate,
        "kupiec_lr": lr,
        "kupiec_p": kp,
        "dgp": dgp_label,
        "claim": "research_metric_only",
    }
    if dates_test is not None:
        from quant_fund.metrics.inference import grouped_mean_tstat

        date_arr = np.asarray(dates_test).reshape(-1)
        if date_arr.size != y_t.size:
            raise ValueError("dates_test must align with y_test")
        mean_miss, t_miss, p_miss, n_dates = grouped_mean_tstat(
            1.0 - covered(y_t, plo, phi), date_arr, target=float(alpha)
        )
        out["date_clustered_miss_rate"] = mean_miss
        out["date_clustered_t"] = t_miss
        out["date_clustered_p"] = p_miss
        out["date_clustered_n_dates"] = float(n_dates)
    if dgp_label == "fixture":
        out["seed"] = float(seed)
    return out
