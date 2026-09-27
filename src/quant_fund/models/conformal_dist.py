"""Conformalized Student-t distribution head (ULTRAPLAN P1.4, ``dip_conf_t``).

Split-conformal recalibration of a Hansen (1994) skew-t base in the style of
Romano, Patterson & Candès (2019, CQR).  ``fit`` receives rows in panel
(date-major) order and treats row order as time order: the skew-t base is
fit by maximum likelihood on the leading two-thirds of the fit window and
the per-tau additive shift is computed on the trailing slice, so calibration
is causal within the fold — no row contributes both to the base parameters
and to its own conformity score.

For each tau the conformity scores on the calibration slice are
``e_i = y_i - q_tau_base`` and the shift ``s_tau`` is the
``ceil((n_cal + 1) * tau)``-th order statistic of ``{e_i}`` (clipped to the
sample when the level is unattainable). Because this head is unconditional,
the skew-t base quantile cancels in that addition: predictions are the
trailing calibration slice's empirical quantiles. The order statistic gives
the stated coverage on that same calibration slice; it does not guarantee
coverage on a particular future sample. Quantiles are rearranged monotone
and tiled at predict time (``x`` is ignored). Fail-closed on degenerate input.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.scoring import rearrange_quantiles
from quant_fund.models.base import JoblibMixin, ModelMeta
from quant_fund.models.skew_t import skew_t_fit, skew_t_ppf

Array = NDArray[np.float64]

MIN_OBS = 60


class ConformalTDistribution(JoblibMixin):
    """Unconditional calibration quantiles under the legacy ``conf_t`` name."""

    def __init__(self, taus: list[float]) -> None:
        self.taus = taus
        self.params_: dict[str, float] | None = None
        self.shifts_: Array | None = None
        self.n_cal_ = 0
        self.q_: Array | None = None

    def fit(
        self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any
    ) -> ConformalTDistribution:
        yy = np.asarray(y, dtype=float).reshape(-1)
        yy = yy[np.isfinite(yy)]
        if yy.size < MIN_OBS:
            raise ValueError("ConformalTDistribution requires >= 60 finite observations")
        tt = np.asarray(self.taus, dtype=float)
        if tt.ndim != 1 or tt.size == 0 or not bool(np.all((tt > 0.0) & (tt < 1.0))):
            raise ValueError("taus must be a non-empty vector inside (0, 1)")
        cut = (2 * yy.size) // 3
        y_base, y_cal = yy[:cut], yy[cut:]
        params = skew_t_fit(y_base)
        q_base = np.array(
            [skew_t_ppf(t, params["nu"], params["lam"], params["mu"], params["sigma"]) for t in tt]
        )
        n = int(y_cal.size)
        shifts = np.empty(tt.size)
        for j in range(tt.size):
            e = y_cal - q_base[j]
            k = min(max(int(np.ceil((n + 1) * tt[j])), 1), n)
            shifts[j] = float(np.partition(e, k - 1)[k - 1])
        self.params_ = params
        self.shifts_ = shifts
        self.n_cal_ = n
        self.q_ = rearrange_quantiles((q_base + shifts).reshape(1, -1))[0]
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.q_ is None:
            raise RuntimeError("distribution model has not been fitted")
        return np.tile(self.q_, (x.shape[0], 1))

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="distribution",
            name="conf_t",
            version="v1",
            extra={
                "n_cal": self.n_cal_,
                **{k: float(v) for k, v in (self.params_ or {}).items() if k != "loglik"},
            },
        )
