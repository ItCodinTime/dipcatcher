"""LightGBM quantile head on the full causal feature matrix (dip_lgbm_q2, P1.8)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.scoring import rearrange_quantiles
from quant_fund.models.base import JoblibMixin, ModelMeta
from quant_fund.models.ranking import _finite

MIN_OBS = 50


class LGBMQ2Distribution(JoblibMixin):
    """Per-tau LightGBM quantile boosters over the full causal feature set.

    x is already the PIT causal design (realized-vol term structure, OHLC
    range, amount; <=30 features upstream); unlike ``TreeQuantileDistribution``
    this v2 head trains on every column — no feature dropping. Predicted
    quantiles are rearranged per row for monotonicity (Chernozhukov,
    Fernandez-Val, Galichon 2010). Row order is irrelevant to the
    estimator; deterministic under ``seed``. Warmup disclosure (n_train,
    n_features, n_estimators) is reported in ``metadata().extra``.
    """

    def __init__(self, taus: list[float], seed: int = 42, n_estimators: int = 150) -> None:
        self.taus = taus
        self.seed = seed
        self.n_estimators = n_estimators
        self.models_: list[Any] = []
        self.n_train_ = 0
        self.n_features_ = 0
        self._fitted = False

    def fit(
        self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any
    ) -> LGBMQ2Distribution:
        from lightgbm import LGBMRegressor

        xx = np.asarray(x, dtype=float)
        yy = np.asarray(y, dtype=float).reshape(-1)
        if xx.ndim != 2 or xx.shape[1] < 1:
            raise ValueError("LGBMQ2Distribution requires x with >= 1 feature column")
        if yy.shape[0] != xx.shape[0]:
            raise ValueError("x and y must have the same number of rows")
        xx, yy, _ = _finite(xx, yy)
        if xx.shape[0] < MIN_OBS:
            raise ValueError(f"LGBMQ2Distribution requires >= {MIN_OBS} finite observations")
        if float(np.std(yy)) <= 0.0:
            raise ValueError("LGBMQ2Distribution requires nonzero variance in y")
        self.models_ = [
            LGBMRegressor(
                objective="quantile",
                alpha=float(tau),
                n_estimators=self.n_estimators,
                num_leaves=31,
                learning_rate=0.05,
                min_child_samples=10,
                subsample=0.9,
                subsample_freq=1,
                colsample_bytree=0.9,
                n_jobs=1,
                random_state=self.seed,
                verbosity=-1,
            ).fit(xx, yy)
            for tau in self.taus
        ]
        self.n_train_ = int(xx.shape[0])
        self.n_features_ = int(xx.shape[1])
        self._fitted = True
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self._fitted or not self.models_:
            raise RuntimeError("distribution model has not been fitted")
        xx = np.asarray(x, dtype=float)
        if xx.ndim != 2:
            raise ValueError("LGBMQ2Distribution requires a 2d feature matrix")
        xx = np.where(np.isfinite(xx), xx, 0.0)
        q = np.column_stack([m.predict(xx) for m in self.models_])
        return np.asarray(rearrange_quantiles(q), dtype=np.float64)

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="distribution",
            name="lgbm_q2",
            version="v1",
            extra={
                "n_train": self.n_train_,
                "n_features": self.n_features_,
                "n_estimators": self.n_estimators,
                "seed": self.seed,
            },
        )
