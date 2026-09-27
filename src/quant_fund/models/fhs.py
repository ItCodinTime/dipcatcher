"""Filtered historical simulation with a skew-t residual law (dip_fhs_skew).

FHS (Barone-Adesi et al. 1999) standardizes returns by a GARCH sigma path
and rescales the standardized residuals by the one-step sigma. This head
replaces the empirical residual quantiles with a Hansen skew-t MLE fit
(``models/skew_t.py``), capturing residual asymmetry that symmetric FHS
misses.

Sigma filter: fixed-coefficient GJR-GARCH(1,1) recursion

    s2_t = w + (a + g 1{e_{t-1} < 0}) e^2_{t-1} + b s2_{t-1},

with omega moment-matched so unconditional E[s2] equals the sample
variance (no QMLE loop — deterministic): w = (1 - b) v - a v - g v_neg
with v = E[e^2], v_neg = E[e^2 1{e<0}].

Predictions are unconditional: q_tau = mu + sigma_end * F^{-1}(tau),
where sigma_end is the one-step GJR variance forecast at the end of the
fit window (no mean-reversion shrinkage — an honest multi-step horizon is
not modeled). Fit rows are assumed to arrive in time order (panel order is
date-major); the sigma path is a recursion over row order, so the
assumption is load-bearing.

Fail-closed: requires >= 60 finite observations, positive unconditional
variance, and a finite positive sigma path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.scoring import rearrange_quantiles
from quant_fund.models.base import JoblibMixin, ModelMeta
from quant_fund.models.skew_t import skew_t_fit, skew_t_ppf

MIN_OBS = 60
# Conventional GJR(1,1) coefficients on returns: persistence a + g/2 + b = 0.95.
_GJR_A = 0.02
_GJR_G = 0.10
_GJR_B = 0.88


def _gjr_omega(e: NDArray[np.float64]) -> float:
    """Moment-matched omega: E[s2] = w + a E[e^2] + g E[e^2 1{e<0}] + b E[s2]."""
    v = float(np.mean(e * e))
    v_neg = float(np.mean(e * e * (e < 0.0)))
    return v * (1.0 - _GJR_B) - _GJR_A * v - _GJR_G * v_neg


def _gjr_sigma_path(e: NDArray[np.float64], w: float) -> NDArray[np.float64]:
    """Fixed-coefficient GJR variance path; fail-closed on degenerate path."""
    s2 = np.empty(e.size)
    with np.errstate(over="ignore", invalid="ignore"):
        s2[0] = float(np.mean(e * e))
        for t in range(1, e.size):
            prev = e[t - 1]
            s2[t] = w + (_GJR_A + _GJR_G * float(prev < 0.0)) * prev * prev + _GJR_B * s2[t - 1]
    if not np.all(np.isfinite(s2)) or np.any(s2 <= 0.0):
        raise ValueError("GJR sigma path is degenerate (non-finite or non-positive)")
    return np.asarray(np.sqrt(s2))


class FhsSkewDistribution(JoblibMixin):
    """FHS head: GJR-filtered residuals -> skew-t law -> one-step rescale."""

    def __init__(self, taus: list[float]) -> None:
        self.taus = taus
        self.mu_ = 0.0
        self.sigma_end_ = 0.0
        self.params_: dict[str, float] | None = None
        self.q_: NDArray[np.float64] | None = None

    def fit(
        self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any
    ) -> FhsSkewDistribution:
        yy = np.asarray(y, dtype=float).reshape(-1)
        yy = yy[np.isfinite(yy)]
        if yy.size < MIN_OBS:
            raise ValueError("FhsSkewDistribution requires >= 60 finite observations")
        self.mu_ = float(yy.mean())
        e = yy - self.mu_
        if float(np.var(e)) <= 0.0:
            raise ValueError("FhsSkewDistribution requires positive variance")
        w = _gjr_omega(e)
        sigma = _gjr_sigma_path(e, w)
        z = e / sigma
        self.params_ = skew_t_fit(z)
        # One-step GJR variance forecast past the window end.
        s2_end = (
            w + (_GJR_A + _GJR_G * float(e[-1] < 0.0)) * e[-1] * e[-1] + _GJR_B * sigma[-1] ** 2
        )
        if not np.isfinite(s2_end) or s2_end <= 0.0:
            raise ValueError("one-step sigma forecast is degenerate")
        self.sigma_end_ = float(np.sqrt(s2_end))
        p = self.params_
        q = np.array(
            [
                self.mu_ + self.sigma_end_ * skew_t_ppf(t, p["nu"], p["lam"], p["mu"], p["sigma"])
                for t in self.taus
            ]
        )
        self.q_ = rearrange_quantiles(q.reshape(1, -1))[0]
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.q_ is None:
            raise RuntimeError("distribution model has not been fitted")
        return np.tile(self.q_, (x.shape[0], 1))

    def metadata(self) -> ModelMeta:
        extra = {k: float(v) for k, v in (self.params_ or {}).items() if k != "loglik"}
        extra["sigma_end"] = self.sigma_end_
        return ModelMeta(family="distribution", name="fhs_skew", version="v1", extra=extra)
