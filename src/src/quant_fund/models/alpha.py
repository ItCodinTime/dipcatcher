"""Expected excess return / residual alpha."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression

from quant_fund.models.base import JoblibMixin, ModelMeta
from quant_fund.models.ranking import RidgeRanker


class HistoricalMeanAlpha(JoblibMixin):
    def __init__(self) -> None:
        self.mean_ = 0.0

    def fit(
        self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any
    ) -> HistoricalMeanAlpha:
        yy = y[np.isfinite(y)]
        self.mean_ = float(np.mean(yy)) if yy.size else 0.0
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.full(x.shape[0], self.mean_)

    def metadata(self) -> ModelMeta:
        return ModelMeta(family="alpha", name="historical_mean", version="v1")


class RidgeAlpha(RidgeRanker):
    def metadata(self) -> ModelMeta:
        return ModelMeta(family="alpha", name="ridge", version="v1", extra={"alpha": self.alpha})


def rolling_beta(
    asset_ret: NDArray[np.float64], mkt_ret: NDArray[np.float64], window: int = 60
) -> NDArray[np.float64]:
    n = len(asset_ret)
    out = np.full(n, np.nan)
    for i in range(window, n + 1):
        a = asset_ret[i - window : i]
        m = mkt_ret[i - window : i]
        if np.std(m) == 0:
            continue
        var = np.var(m)
        out[i - 1] = np.cov(a, m, ddof=0)[0, 1] / var
    return out


def residualize(
    asset_ret: NDArray[np.float64],
    factors: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """OLS residual of asset on factor matrix (n, k). Returns residual, betas, intercept."""
    mask = np.isfinite(asset_ret) & np.isfinite(factors).all(axis=1)
    y = asset_ret[mask]
    x = factors[mask]
    lr = LinearRegression().fit(x, y)
    resid = np.full_like(asset_ret, np.nan)
    resid[mask] = y - lr.predict(x)
    return resid, lr.coef_.astype(float), float(lr.intercept_)
