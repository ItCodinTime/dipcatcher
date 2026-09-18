"""Split CQR and adaptive conformal inference wrappers.

Wraps existing quantile / point predictors. Does not reimplement QR.
Romano, Patterson, Candès (2019); Gibbs & Candès (2021).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.conformal import (
    conformal_quantile,
    covered,
    cqr_scores,
    expand_interval,
    onesided_scores,
)
from quant_fund.metrics.cross_section import _date_keys
from quant_fund.models.base import JoblibMixin, ModelMeta

Array = NDArray[np.float64]


class SplitCQR(JoblibMixin):
    """Calibrate a residual quantile on a held-out window, then expand bands."""

    def __init__(self, alpha: float = 0.10) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)
        self.qhat = 0.0
        self.scores_: Array | None = None

    def calibrate(self, y: Array, lower: Array, upper: Array) -> SplitCQR:
        s = cqr_scores(y, lower, upper)
        self.scores_ = s
        self.qhat = conformal_quantile(s, self.alpha)
        return self

    def predict_sets(self, lower: Array, upper: Array) -> tuple[Array, Array]:
        return expand_interval(lower, upper, self.qhat)

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="conformal",
            name="split_cqr",
            version="v1",
            extra={"alpha": self.alpha, "qhat": self.qhat},
        )


class SplitOneSided(JoblibMixin):
    """One-sided conformal upper bound. Used for loss/VaR tails."""

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)
        self.qhat = 0.0

    def calibrate(self, y: Array, bound: Array) -> SplitOneSided:
        self.qhat = conformal_quantile(onesided_scores(y, bound), self.alpha)
        return self

    def predict_bound(self, bound: Array) -> Array:
        return np.asarray(bound, dtype=float) + self.qhat

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="conformal",
            name="split_onesided",
            version="v1",
            extra={"alpha": self.alpha, "qhat": self.qhat},
        )


@dataclass
class ACIPath:
    lower: Array
    upper: Array
    covered: Array
    alpha_t: Array
    qhat_t: Array
    dates: list[str] = field(default_factory=list)


class AdaptiveConformal:
    """Gibbs–Candès ACI on residual scores, updated once per timestamp.

    alpha_{t+1} = clip(alpha_t + gamma * (alpha - err_t), eps, 1-eps).
    After a miss, alpha_t falls and sets widen.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        gamma: float = 0.05,
        score_window: int | None = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if gamma <= 0:
            raise ValueError("gamma must be > 0")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.score_window = score_window
        self.alpha_t = float(alpha)
        self.scores: list[float] = []

    def initialize(self, y: Array, lower: Array, upper: Array, scale: Array | None = None) -> None:
        s = cqr_scores(y, lower, upper, scale)
        self.scores = [float(v) for v in s[np.isfinite(s)]]
        self.alpha_t = self.alpha

    def qhat(self) -> float:
        if not self.scores:
            return 0.0
        return conformal_quantile(np.asarray(self.scores, dtype=float), self.alpha_t)

    def _append_scores(self, scores: Array) -> None:
        self.scores.extend(float(v) for v in np.asarray(scores, dtype=float) if np.isfinite(v))
        if self.score_window is not None and len(self.scores) > self.score_window:
            self.scores = self.scores[-int(self.score_window) :]

    def update(self, err: float) -> None:
        nxt = self.alpha_t + self.gamma * (self.alpha - float(err))
        self.alpha_t = float(np.clip(nxt, 1e-3, 1.0 - 1e-3))

    def run(
        self,
        y: Array,
        lower: Array,
        upper: Array,
        dates: NDArray[Any] | list[object],
        scale: Array | None = None,
    ) -> ACIPath:
        """Walk dates in order. Calibration scores must already be in `initialize`."""
        y = np.asarray(y, dtype=float)
        lo = np.asarray(lower, dtype=float)
        hi = np.asarray(upper, dtype=float)
        sc = None if scale is None else np.asarray(scale, dtype=float)
        keys = _date_keys(dates)
        order = sorted(set(keys))
        out_lo = np.full(y.size, np.nan)
        out_hi = np.full(y.size, np.nan)
        out_cov = np.full(y.size, np.nan)
        alphas: list[float] = []
        qhats: list[float] = []
        kept: list[str] = []
        for key in order:
            idx = np.array([k == key for k in keys], dtype=bool)
            qh = self.qhat()
            sc_i = None if sc is None else sc[idx]
            lo_c, hi_c = expand_interval(lo[idx], hi[idx], qh, sc_i)
            cov = covered(y[idx], lo_c, hi_c)
            out_lo[idx] = lo_c
            out_hi[idx] = hi_c
            out_cov[idx] = cov
            err = float(1.0 - np.nanmean(cov)) if cov.size else 1.0
            self._append_scores(cqr_scores(y[idx], lo[idx], hi[idx], sc_i))
            self.update(err)
            alphas.append(self.alpha_t)
            qhats.append(qh)
            kept.append(key)
        return ACIPath(
            lower=out_lo,
            upper=out_hi,
            covered=out_cov,
            alpha_t=np.asarray(alphas, dtype=float),
            qhat_t=np.asarray(qhats, dtype=float),
            dates=kept,
        )


class MondrianCQR(JoblibMixin):
    """Separate residual quantile per PIT-safe stratum (Vovk Mondrian conformal)."""

    def __init__(self, alpha: float = 0.10, min_count: int = 12) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)
        self.min_count = int(min_count)
        self.qhat: dict[str, float] = {}
        self.global_qhat = 0.0

    def calibrate(
        self,
        y: Array,
        lower: Array,
        upper: Array,
        labels: Array,
        scale: Array | None = None,
    ) -> MondrianCQR:
        y = np.asarray(y, dtype=float)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        lab = np.asarray(labels)
        self.global_qhat = conformal_quantile(cqr_scores(y, lower, upper, scale), self.alpha)
        self.qhat = {}
        for g in np.unique(lab):
            sel = lab == g
            if int(sel.sum()) < self.min_count:
                continue
            sc = None if scale is None else np.asarray(scale, dtype=float)[sel]
            self.qhat[str(g)] = conformal_quantile(
                cqr_scores(y[sel], lower[sel], upper[sel], sc), self.alpha
            )
        return self

    def _row_qhat(self, labels: Array) -> Array:
        return np.asarray(
            [self.qhat.get(str(g), self.global_qhat) for g in np.asarray(labels)],
            dtype=float,
        )

    def predict_sets(
        self, lower: Array, upper: Array, labels: Array, scale: Array | None = None
    ) -> tuple[Array, Array]:
        return expand_interval(lower, upper, self._row_qhat(labels), scale)

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="conformal",
            name="mondrian_cqr",
            version="v1",
            extra={"alpha": self.alpha, "qhat": dict(self.qhat), "global_qhat": self.global_qhat},
        )


class MondrianACI:
    """Per-stratum ACI. Each group has its own alpha_t and residual scores."""

    def __init__(
        self,
        alpha: float = 0.10,
        gamma: float = 0.05,
        score_window: int | None = None,
        min_count: int = 12,
    ) -> None:
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.score_window = score_window
        self.min_count = int(min_count)
        self.fallback = AdaptiveConformal(alpha, gamma, score_window)
        self.groups: dict[str, AdaptiveConformal] = {}

    def initialize(
        self,
        y: Array,
        lower: Array,
        upper: Array,
        labels: Array,
        scale: Array | None = None,
    ) -> None:
        lab = np.asarray(labels)
        self.fallback.initialize(y, lower, upper, scale)
        self.groups = {}
        for g in np.unique(lab):
            sel = lab == g
            if int(sel.sum()) < self.min_count:
                continue
            aci = AdaptiveConformal(self.alpha, self.gamma, self.score_window)
            sc = None if scale is None else np.asarray(scale, dtype=float)[sel]
            aci.initialize(y[sel], lower[sel], upper[sel], sc)
            self.groups[str(g)] = aci

    def _qhat_for(self, label: str) -> float:
        g = self.groups.get(label)
        if g is None or len(g.scores) < self.min_count:
            return self.fallback.qhat()
        return g.qhat()

    def run(
        self,
        y: Array,
        lower: Array,
        upper: Array,
        dates: NDArray[Any] | list[object],
        labels: Array,
        scale: Array | None = None,
    ) -> ACIPath:
        y = np.asarray(y, dtype=float)
        lo = np.asarray(lower, dtype=float)
        hi = np.asarray(upper, dtype=float)
        lab = np.asarray(labels)
        sc = None if scale is None else np.asarray(scale, dtype=float)
        keys = _date_keys(dates)
        order = sorted(set(keys))
        out_lo = np.full(y.size, np.nan)
        out_hi = np.full(y.size, np.nan)
        out_cov = np.full(y.size, np.nan)
        alphas: list[float] = []
        qhats: list[float] = []
        kept: list[str] = []
        for key in order:
            idx = np.array([k == key for k in keys], dtype=bool)
            q_row = np.asarray([self._qhat_for(str(g)) for g in lab[idx]], dtype=float)
            sc_i = None if sc is None else sc[idx]
            lo_c, hi_c = expand_interval(lo[idx], hi[idx], q_row, sc_i)
            cov = covered(y[idx], lo_c, hi_c)
            out_lo[idx] = lo_c
            out_hi[idx] = hi_c
            out_cov[idx] = cov
            err = float(1.0 - np.nanmean(cov)) if cov.size else 1.0
            self.fallback._append_scores(cqr_scores(y[idx], lo[idx], hi[idx], sc_i))
            self.fallback.update(err)
            lab_i = lab[idx]
            for name, aci in self.groups.items():
                sel = np.array([str(g) == name for g in lab_i], dtype=bool)
                if not bool(sel.any()):
                    continue
                sc_g = None if sc_i is None else sc_i[sel]
                aci._append_scores(cqr_scores(y[idx][sel], lo[idx][sel], hi[idx][sel], sc_g))
                err_g = float(1.0 - np.nanmean(cov[sel]))
                aci.update(err_g)
            alphas.append(self.fallback.alpha_t)
            qhats.append(float(np.mean(q_row)) if q_row.size else 0.0)
            kept.append(key)
        return ACIPath(
            lower=out_lo,
            upper=out_hi,
            covered=out_cov,
            alpha_t=np.asarray(alphas, dtype=float),
            qhat_t=np.asarray(qhats, dtype=float),
            dates=kept,
        )
