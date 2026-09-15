"""Volatility forecasts: rolling, EWMA, GARCH, HAR-RV, trees."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression

from quant_fund.models.base import JoblibMixin, ModelMeta
from quant_fund.models.ranking import _finite


class RollingVol(JoblibMixin):
    def __init__(self, window: int = 20) -> None:
        self.window = window

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> RollingVol:
        return self

    def predict_from_returns(self, log_returns: NDArray[np.float64]) -> NDArray[np.float64]:
        r = np.asarray(log_returns, dtype=float)
        out = np.full_like(r, np.nan)
        for i in range(self.window, r.size + 1):
            out[i - 1] = np.std(r[i - self.window : i], ddof=1)
        return out

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        # last column assumed to be trailing vol already
        return x[:, -1]

    def metadata(self) -> ModelMeta:
        return ModelMeta(family="volatility", name="rolling", version="v1")


def ewma_variance(log_returns: NDArray[np.float64], lam: float = 0.94) -> NDArray[np.float64]:
    r = np.asarray(log_returns, dtype=float)
    var = np.empty_like(r)
    var[0] = r[0] ** 2
    for t in range(1, r.size):
        var[t] = lam * var[t - 1] + (1.0 - lam) * r[t - 1] ** 2
    return var


class EWMAVol(JoblibMixin):
    def __init__(self, lam: float = 0.94) -> None:
        self.lam = lam

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> EWMAVol:
        return self

    def predict_from_returns(self, log_returns: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.sqrt(ewma_variance(log_returns, self.lam))

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return x[:, -1]

    def metadata(self) -> ModelMeta:
        return ModelMeta(family="volatility", name="ewma", version="v1", extra={"lambda": self.lam})


class GARCHVol(JoblibMixin):
    def __init__(self) -> None:
        self.result: Any = None
        self.last_sigma: float = 0.01

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> GARCHVol:
        from arch import arch_model

        r = y[np.isfinite(y)] * 100.0  # percent for numerical stability
        if r.size < 50:
            self.last_sigma = float(np.std(r, ddof=1) / 100.0) if r.size > 1 else 0.01
            return self
        am = arch_model(r, vol="GARCH", p=1, o=0, q=1, dist="normal", mean="Constant")
        self.result = am.fit(disp="off")
        # arch returns conditional_volatility as sigma, not variance.
        self.last_sigma = float(self.result.conditional_volatility.iloc[-1] / 100.0)
        a = float(self.result.params.get("alpha[1]", 0.0))
        b = float(self.result.params.get("beta[1]", 0.0))
        if a + b >= 1:
            self.last_sigma = float(np.std(r) / 100.0)
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.result is None:
            return np.full(x.shape[0], self.last_sigma)
        f = self.result.forecast(horizon=1)
        v = float(f.variance.values[-1, 0]) / 10000.0
        return np.full(x.shape[0], np.sqrt(max(v, 1e-16)))

    def metadata(self) -> ModelMeta:
        return ModelMeta(family="volatility", name="garch11", version="v1")


class HARVol(JoblibMixin):
    def __init__(self, use_log: bool = True) -> None:
        self.use_log = use_log
        self.model = LinearRegression()
        self.fitted = False

    @staticmethod
    def har_design(rv: NDArray[np.float64]) -> NDArray[np.float64]:
        """Build HAR regressors available immediately before each target time.

        Row ``t`` forecasts realized volatility at ``t`` using ``rv[t-1]`` and
        trailing averages ending at ``t-1``.  The prior implementation included
        ``rv[t]`` in every row, making in-sample forecasts contemporaneously
        leak their target.
        """
        values = np.asarray(rv, dtype=float).reshape(-1)
        n = values.size
        lagged = np.full(n, np.nan)
        if n > 1:
            lagged[1:] = values[:-1]

        def trailing_mean(window: int) -> NDArray[np.float64]:
            result = np.full(n, np.nan)
            for t in range(window, n):
                result[t] = np.mean(lagged[t - window + 1 : t + 1])
            return result

        return np.column_stack([np.ones(n), lagged, trailing_mean(5), trailing_mean(22)])

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> HARVol:
        rv = np.asarray(y, dtype=float)
        target = np.log(np.clip(rv, 1e-12, None)) if self.use_log else rv
        des = self.har_design(rv)
        mask = np.isfinite(des).all(axis=1) & np.isfinite(target)
        if mask.sum() < 10:
            return self
        self.model.fit(des[mask], target[mask])
        self.fitted = True
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.fitted:
            return np.full(x.shape[0], np.nan)
        # x columns are already HAR features if provided; else treat last col as rv
        pred = self.model.predict(x)
        if self.use_log:
            pred = np.exp(pred)
        return pred

    def metadata(self) -> ModelMeta:
        return ModelMeta(family="volatility", name="har_rv", version="v1")


class TreeVol(JoblibMixin):
    def __init__(self, backend: str = "lightgbm", seed: int = 42) -> None:
        self.backend = backend
        if backend == "xgboost":
            from xgboost import XGBRegressor

            self.model = XGBRegressor(
                n_estimators=80, max_depth=3, n_jobs=1, random_state=seed, verbosity=0
            )
        else:
            from lightgbm import LGBMRegressor

            self.model = LGBMRegressor(
                n_estimators=80, num_leaves=15, random_state=seed, verbosity=-1
            )

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> TreeVol:
        xx, yy, _ = _finite(x, y)
        self.model.fit(xx, yy)
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(self.model.predict(x), dtype=float)

    def metadata(self) -> ModelMeta:
        return ModelMeta(family="volatility", name=f"{self.backend}_vol", version="v1")
