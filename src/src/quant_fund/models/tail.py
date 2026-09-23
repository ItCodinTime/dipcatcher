"""Tail risk: historical / Gaussian / distribution-derived VaR-ES and drawdown probs."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from quant_fund.metrics.risk import (
    gaussian_es,
    gaussian_var,
    historical_es,
    historical_var,
    losses_from_returns,
)
from quant_fund.models.base import JoblibMixin, ModelMeta
from quant_fund.models.ranking import _finite


class HistoricalTail(JoblibMixin):
    def __init__(self, alpha: float = 0.95) -> None:
        self.alpha = alpha
        self.losses: NDArray[np.float64] | None = None

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> HistoricalTail:
        self.losses = losses_from_returns(y)
        return self

    def predict_var_es(self) -> tuple[float, float]:
        if self.losses is None:
            raise RuntimeError("historical tail model has not been fitted")
        return historical_var(self.losses, self.alpha), historical_es(self.losses, self.alpha)

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        var, es = self.predict_var_es()
        return np.tile(np.array([var, es]), (x.shape[0], 1))

    def metadata(self) -> ModelMeta:
        return ModelMeta(family="tail", name="historical", version="v1")


class ScaledHistoricalTail(JoblibMixin):
    """Historical VaR/ES on losses / scale, then multiply by current scale."""

    def __init__(self, alpha: float = 0.95) -> None:
        self.alpha = alpha
        self.z_var = 0.0
        self.z_es = 0.0

    def fit(
        self,
        y: NDArray[np.float64],
        scale: NDArray[np.float64],
        **kwargs: Any,
    ) -> ScaledHistoricalTail:
        yy = np.asarray(y, dtype=float).reshape(-1)
        sc = np.maximum(np.asarray(scale, dtype=float).reshape(-1), 1e-8)
        if yy.size != sc.size:
            raise ValueError("y and scale must have the same length")
        losses = (-yy) / sc
        losses = losses[np.isfinite(losses)]
        if losses.size == 0:
            self.z_var = 0.0
            self.z_es = 0.0
            return self
        self.z_var = historical_var(losses, self.alpha)
        self.z_es = historical_es(losses, self.alpha)
        return self

    def predict_var_es(
        self, scale: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        sc = np.maximum(np.asarray(scale, dtype=float).reshape(-1), 1e-8)
        return self.z_var * sc, self.z_es * sc

    def metadata(self) -> ModelMeta:
        return ModelMeta(family="tail", name="scaled_historical", version="v1")


class GaussianTail(HistoricalTail):
    def predict_var_es(self) -> tuple[float, float]:
        if self.losses is None:
            raise RuntimeError("Gaussian tail model has not been fitted")
        return gaussian_var(self.losses, self.alpha), gaussian_es(self.losses, self.alpha)

    def metadata(self) -> ModelMeta:
        return ModelMeta(family="tail", name="gaussian", version="v1")


class DrawdownClassifier(JoblibMixin):
    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=500)

    def fit(
        self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any
    ) -> DrawdownClassifier:
        xx, yy, _ = _finite(x, y)
        yy = (yy > 0.5).astype(int)
        if len(np.unique(yy)) < 2:
            return self
        self.model.fit(xx, yy)
        return self

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.where(np.isfinite(x), x, 0.0)
        if not hasattr(self.model, "coef_"):
            return np.full(x.shape[0], 0.05)
        return np.asarray(self.model.predict_proba(x)[:, 1], dtype=np.float64)

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.predict_proba(x)

    def metadata(self) -> ModelMeta:
        return ModelMeta(family="tail", name="drawdown_logit", version="v1")


def stylized_stress(
    equity_shock: float = -0.10, vol_mult: float = 2.0, corr: float = 0.9
) -> dict[str, float | str]:
    """Hypothetical shocks. Not a named historical crisis replay."""
    return {
        "equity_shock": equity_shock,
        "vol_multiplier": vol_mult,
        "correlation": corr,
        "note": "stylized_hypothetical",
    }
