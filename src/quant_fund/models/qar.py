"""Quantile autoregression (Koenker & Xiao 2006).

QAR(p): Q_{y_t}(tau | F_{t-1}) = a0(tau) + sum_j a_j(tau) y_{t-j}.
Each quantile is estimated by Koenker-Bassett quantile regression on
lagged values (delegated to ``metrics.regression.quantile_regression``).

Unit-root diagnostics across the quantile surface: the persistence
estimate a1(tau) may exceed 1 in the lower tail (local explosion)
even when the median process is stationary -- this is the asymmetric
adjustment phenomenon the QAR paper documents.

``QARDistribution`` packages the AR(1) special case as a distribution
head for the training pipeline.

Fail-closed: tau outside (0,1), insufficient data, non-finite y.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import QuantileRegressor

from quant_fund.metrics.regression import quantile_regression
from quant_fund.metrics.scoring import rearrange_quantiles
from quant_fund.models.base import JoblibMixin, ModelMeta

Array = NDArray[np.float64]


def _check(y: Array, p: int) -> Array:
    yy = np.asarray(y, dtype=float).ravel()
    if yy.size < 4 * p + 40 or not np.isfinite(yy).all():
        raise ValueError("insufficient or non-finite data")
    if p < 1:
        raise ValueError("p must be >= 1")
    return yy


def qar_fit(y: Array, p: int, taus: Array) -> dict[str, Array]:
    """Fit QAR(p) at each tau. Returns coef matrix (len(taus), p+1)."""
    yy = _check(y, p)
    tt = np.asarray(taus, dtype=float).ravel()
    if tt.size == 0 or (tt <= 0).any() or (tt >= 1).any():
        raise ValueError("taus must lie in (0, 1)")
    n = yy.size
    xl = np.column_stack([yy[p - i - 1 : n - i - 1] for i in range(p)])
    yt = yy[p:]
    coefs = np.empty((tt.size, p + 1))
    for i, tau in enumerate(tt):
        out = quantile_regression(xl, yt, float(tau))
        coefs[i] = np.asarray(out["beta"], dtype=float)
    x = np.column_stack([np.ones(n - p), xl])
    return {"coef": coefs, "taus": tt, "x": x, "y": yt}


def qar_summary(fit: dict[str, Array]) -> dict[str, Array]:
    """Persistence profile a1(tau) and a Kolmogorov-Smirnov-style
    uniformity diagnostic on the fitted quantile surface."""
    coef = np.asarray(fit["coef"], dtype=float)
    taus = np.asarray(fit["taus"], dtype=float)
    x = np.asarray(fit["x"], dtype=float)
    y = np.asarray(fit["y"], dtype=float)
    # fraction of realizations below their fitted conditional quantile
    # should equal tau on average
    below = np.zeros(taus.size)
    for i in range(taus.size):
        q = x @ coef[i]
        below[i] = float((y <= q).mean())
    return {
        "persistence_a1": coef[:, 1],
        "coverage": below,
        "coverage_dev": np.abs(below - taus),
    }


class QARDistribution(JoblibMixin):
    """QAR(1) challenger head (Koenker & Xiao 2006).

    Per tau, fit ``q_tau(t) = a_tau + b_tau * y_{t-1}`` by
    Koenker-Bassett quantile regression on consecutive observations from
    one chronological series. The caller must supply a single-security,
    ordered series; this standalone model does not accept a mixed panel.
    Non-finite rows fail closed instead of silently bridging a missing gap.

    ``predict`` returns only the one-step-ahead conditional quantiles at
    the last fitted ``y``. It rejects multirow requests because subsequent
    origins require observations that are unavailable at fit time.
    """

    MIN_OBS = 30

    def __init__(self, taus: list[float]) -> None:
        self.taus = taus
        self.coef_: NDArray[np.float64] | None = None  # (n_taus, 2) [a, b]
        self.y_last_ = 0.0
        self.n_pairs_ = 0

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> QARDistribution:
        yy = np.asarray(y, dtype=float).reshape(-1)
        if yy.size < self.MIN_OBS:
            raise ValueError("QARDistribution requires >= 30 observations")
        if not np.isfinite(yy).all():
            raise ValueError("QARDistribution requires all finite observations")
        if float(np.ptp(yy)) <= 0.0:
            raise ValueError("QARDistribution requires non-constant y")
        tt = np.asarray(self.taus, dtype=float).reshape(-1)
        if tt.size == 0 or not np.isfinite(tt).all() or (tt <= 0.0).any() or (tt >= 1.0).any():
            raise ValueError("taus must lie in (0, 1)")
        y_lag = yy[:-1].reshape(-1, 1)
        y_cur = yy[1:]
        coef = np.empty((tt.size, 2))
        for i, tau in enumerate(tt):
            m = QuantileRegressor(quantile=float(tau), alpha=0.0, solver="highs")
            m.fit(y_lag, y_cur)
            beta = np.concatenate([[m.intercept_], np.asarray(m.coef_, dtype=float)])
            if not np.isfinite(beta).all():
                raise ValueError(f"QAR fit failed at tau={tau}")
            coef[i] = beta
        self.coef_ = coef
        self.y_last_ = float(yy[-1])
        self.n_pairs_ = int(y_cur.size)
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.coef_ is None:
            raise RuntimeError("distribution model has not been fitted")
        if x.ndim != 2 or x.shape[0] != 1:
            raise ValueError("QARDistribution predicts exactly one future observation")
        q = self.coef_[:, 0] + self.coef_[:, 1] * self.y_last_
        q = rearrange_quantiles(q.reshape(1, -1))[0]
        result: NDArray[np.float64] = np.asarray(q, dtype=np.float64).reshape(1, -1)
        return result

    def metadata(self) -> ModelMeta:
        persistence = self.coef_[:, 1].tolist() if self.coef_ is not None else []
        return ModelMeta(
            family="distribution",
            name="qar",
            version="v1",
            extra={"persistence_a1": persistence, "n_pairs": self.n_pairs_},
        )
