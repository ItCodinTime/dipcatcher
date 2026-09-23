"""Realized GARCH volatility using daily Parkinson OHLC measures.

This module intentionally models a *daily* realized measure.  The measure is
computed from split-adjusted daily high/low prices and is not a claim about
intraday or high-frequency realized variance.  The implementation uses a
log-variance plugin with a realized-measure persistence term.  It has the same
causal forecast and density protocol as :class:`GARCHVol`, while keeping the
realized-measure provenance explicit in every diagnostic and forecast.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from .base import JoblibMixin, ModelMeta

REALIZED_GARCH_FAMILY = "realized_garch"
REALIZED_GARCH_MEASURE = "parkinson_daily_ohlc"
REALIZED_GARCH_MULTI_STEP_METHOD = "expected_log_variance_plugin"
_VARIANCE_FLOOR = 1e-12
_MEASURE_FLOOR = 1e-14


def parkinson_daily_variance(
    high: NDArray[np.float64] | list[float], low: NDArray[np.float64] | list[float]
) -> NDArray[np.float64]:
    """Return one-day Parkinson variance for paired daily high/low prices.

    Invalid rows (non-positive, non-finite, or ``high < low``) are undefined
    and returned as ``nan``.  Equal valid prices correctly produce zero.
    """
    highs = np.asarray(high, dtype=float).reshape(-1)
    lows = np.asarray(low, dtype=float).reshape(-1)
    if highs.size != lows.size:
        raise ValueError("high and low must have the same length")
    out = np.full(highs.size, np.nan, dtype=float)
    valid = np.isfinite(highs) & np.isfinite(lows) & (highs > 0.0) & (lows > 0.0) & (highs >= lows)
    if np.any(valid):
        log_range = np.log(highs[valid] / lows[valid])
        out[valid] = (log_range * log_range) / (4.0 * math.log(2.0))
    return out


class RealizedGARCHVol(JoblibMixin):
    """Causal date-level realized GARCH volatility model.

    ``returns`` and ``realized_measure`` are paired observations.  The latent
    log variance is forecast from the historical log Parkinson measure using a
    persistent expected-log-variance plugin.  This is deliberately compact and
    deterministic, making saved artifacts replayable while preserving the key
    realized-GARCH contract: the one-step forecast responds to the last
    realized measure, and longer horizons revert toward its historical level.
    """

    def __init__(
        self,
        *,
        min_obs: int = 50,
        mean: str = "Constant",
        series_scope: str = "univariate_return_series",
        realized_measure: str = REALIZED_GARCH_MEASURE,
    ) -> None:
        if isinstance(min_obs, bool) or not isinstance(min_obs, int) or min_obs < 2:
            raise ValueError("min_obs must be an integer >= 2")
        if mean not in {"Constant", "Zero"}:
            raise ValueError("mean must be 'Constant' or 'Zero'")
        if not isinstance(series_scope, str) or not series_scope.strip():
            raise ValueError("series_scope must be a non-empty string")
        if realized_measure != REALIZED_GARCH_MEASURE:
            raise ValueError(
                "realized_measure must be 'parkinson_daily_ohlc'; "
                "close-to-close and high-frequency measures are not accepted"
            )
        self.min_obs = int(min_obs)
        self.mean = mean
        self.series_scope = series_scope.strip()
        self.realized_measure = realized_measure
        self.intraday_realized_variance = False
        self.variance_family = REALIZED_GARCH_FAMILY

        self.n_obs = 0
        self.fit_status = "unfitted"
        self.converged = False
        self.fallback_reason: str | None = None
        self.last_sigma = 0.01
        self.result: Any = None
        self._mean = 0.0
        self._log_measure_mean = math.log(1e-4)
        self._last_log_measure = self._log_measure_mean
        self._scale = 1.0
        self._persistence = 0.35
        self._returns = np.empty(0, dtype=float)
        self._measures = np.empty(0, dtype=float)

    @staticmethod
    def _as_vector(values: Any, name: str) -> NDArray[np.float64]:
        try:
            out = np.asarray(values, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a numeric one-dimensional array") from exc
        return out

    def _fallback(
        self, returns: NDArray[np.float64], measures: NDArray[np.float64], reason: str
    ) -> RealizedGARCHVol:
        self.result = None
        self.converged = False
        self.fit_status = "fallback"
        self.fallback_reason = reason
        self._returns = returns.copy()
        self._measures = measures.copy()
        if returns.size > 1:
            sigma = float(np.std(returns, ddof=1))
            self.last_sigma = sigma if np.isfinite(sigma) and sigma > 0.0 else 0.01
        elif measures.size:
            sigma = float(np.sqrt(max(np.nanmean(measures), _VARIANCE_FLOOR)))
            self.last_sigma = sigma if np.isfinite(sigma) and sigma > 0.0 else 0.01
        else:
            self.last_sigma = 0.01
        return self

    def fit_returns(
        self, returns: NDArray[np.float64], realized_measure: NDArray[np.float64]
    ) -> RealizedGARCHVol:
        return self.fit(
            np.empty((np.asarray(returns).reshape(-1).size, 0)),
            returns,
            returns=returns,
            realized_measure=realized_measure,
        )

    def fit(self, x: Any, y: Any, **kwargs: Any) -> RealizedGARCHVol:
        """Fit on explicit paired decimal returns and daily Parkinson measures."""
        if "realized_measure" not in kwargs:
            raise ValueError("realized_measure must be supplied explicitly")
        if "returns" not in kwargs:
            raise ValueError("returns= must be supplied explicitly")
        returns = self._as_vector(kwargs["returns"], "returns")
        measures = self._as_vector(kwargs["realized_measure"], "realized_measure")
        if returns.size != measures.size:
            raise ValueError("returns and realized_measure must have the same length")
        valid = np.isfinite(returns) & np.isfinite(measures) & (measures >= 0.0)
        self.n_obs = int(np.count_nonzero(valid))
        paired_returns = returns[valid]
        paired_measures = measures[valid]
        if self.n_obs < self.min_obs:
            return self._fallback(paired_returns, paired_measures, "insufficient_observations")
        if paired_returns.size == 0 or not np.isfinite(paired_returns).all():
            return self._fallback(paired_returns, paired_measures, "invalid_returns")
        self._mean = float(np.mean(paired_returns)) if self.mean == "Constant" else 0.0
        centered = paired_returns - self._mean
        log_measure = np.log(np.maximum(paired_measures, _MEASURE_FLOOR))
        self._log_measure_mean = float(np.mean(log_measure))
        self._last_log_measure = float(log_measure[-1])

        # Use the realized measure as the scale anchor, with a robust ratio to
        # the return variance.  Bounds stop a noisy sample from creating an
        # implausibly large risk overlay while retaining OHLC sensitivity.
        ratios = (centered * centered) / np.maximum(paired_measures, _MEASURE_FLOOR)
        finite_ratios = ratios[np.isfinite(ratios) & (ratios > 0.0)]
        if finite_ratios.size:
            self._scale = float(np.clip(np.median(finite_ratios), 0.05, 20.0))
        else:
            self._scale = 1.0
        self._returns = paired_returns.copy()
        self._measures = paired_measures.copy()
        self.result = {"family": REALIZED_GARCH_FAMILY, "measure": REALIZED_GARCH_MEASURE}
        self.converged = True
        self.fit_status = "fitted"
        self.fallback_reason = None
        self.last_sigma = float(np.sqrt(max(self._one_step_variance(), _VARIANCE_FLOOR)))
        return self

    def _one_step_variance(self) -> float:
        log_state = (
            1.0 - self._persistence
        ) * self._log_measure_mean + self._persistence * self._last_log_measure
        variance = self._scale * math.exp(log_state)
        return float(max(variance, _VARIANCE_FLOOR))

    def forecast(
        self,
        horizon: int = 1,
        *,
        method: str = "analytic",
        simulations: int = 1000,
        quantiles: tuple[float, ...] | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if method not in {"analytic", "simulation", "bootstrap"}:
            raise ValueError("method must be analytic, simulation, or bootstrap")
        if simulations < 1:
            raise ValueError("simulations must be positive")
        if self.fit_status == "unfitted":
            raise ValueError("model must be fitted before forecasting")
        if self.result is None:
            variance = np.full(horizon, max(self.last_sigma**2, _VARIANCE_FLOOR))
        else:
            start = self._one_step_variance()
            long_run = max(self._scale * math.exp(self._log_measure_mean), _VARIANCE_FLOOR)
            # Expected log variance reverts to the historical realized level.
            logs = np.empty(horizon, dtype=float)
            logs[0] = math.log(start)
            for index in range(1, horizon):
                logs[index] = self._persistence * logs[index - 1] + (
                    1.0 - self._persistence
                ) * math.log(long_run)
            variance = np.maximum(np.exp(logs), _VARIANCE_FLOOR)
        out: dict[str, Any] = {
            "variance": variance,
            "sigma": np.sqrt(variance),
            "cumulative_variance": np.cumsum(variance),
            "distribution": "normal",
            "requested_distribution": "normal",
            "horizon": horizon,
            "fit_status": self.fit_status,
            "mean": self._mean_decimal(),
            "realized_measure": self.realized_measure,
            "intraday_realized_variance": False,
            "variance_family": REALIZED_GARCH_FAMILY,
            "multi_step_method": REALIZED_GARCH_MULTI_STEP_METHOD,
        }
        if quantiles is not None:
            levels = np.asarray(quantiles, dtype=float)
            if levels.ndim != 1 or np.any((levels <= 0.0) | (levels >= 1.0)):
                raise ValueError("quantiles must lie strictly between 0 and 1")
            out["quantiles"] = (
                self._mean_decimal() + np.sqrt(variance)[:, None] * norm.ppf(levels)[None, :]
            )
        return out

    def _mean_decimal(self) -> float:
        return float(self._mean)

    def pit(self, returns: Any, sigma: Any | None = None) -> NDArray[np.float64]:
        values = self._as_vector(returns, "returns")
        scales = (
            np.full(values.size, self.last_sigma)
            if sigma is None
            else self._as_vector(sigma, "sigma")
        )
        if values.size != scales.size:
            raise ValueError("returns and sigma must have the same length")
        out = np.full(values.size, np.nan, dtype=float)
        valid = np.isfinite(values) & np.isfinite(scales) & (scales > 0.0)
        out[valid] = norm.cdf((values[valid] - self._mean_decimal()) / scales[valid])
        return np.clip(out, 0.0, 1.0)

    def log_density(self, returns: Any, sigma: Any | None = None) -> NDArray[np.float64]:
        values = self._as_vector(returns, "returns")
        scales = (
            np.full(values.size, self.last_sigma)
            if sigma is None
            else self._as_vector(sigma, "sigma")
        )
        if values.size != scales.size:
            raise ValueError("returns and sigma must have the same length")
        out = np.full(values.size, np.nan, dtype=float)
        valid = np.isfinite(values) & np.isfinite(scales) & (scales > 0.0)
        standardized = (values[valid] - self._mean_decimal()) / scales[valid]
        out[valid] = -0.5 * standardized**2 - np.log(scales[valid]) - 0.5 * math.log(2.0 * math.pi)
        return out

    def predict_from_returns(
        self, returns: Any, realized_measure: Any | None = None
    ) -> NDArray[np.float64]:
        """Compatibility adapter returning current-origin sigma per input row."""
        values = self._as_vector(returns, "returns")
        if realized_measure is not None:
            self.fit_returns(values, realized_measure)
        return np.full(values.size, float(self.forecast(horizon=1)["sigma"][0]))

    def diagnostics(self) -> dict[str, Any]:
        return {
            "n_obs": int(self.n_obs),
            "fit_status": self.fit_status,
            "converged": bool(self.converged),
            "fallback_reason": self.fallback_reason,
            "returns_scale": "decimal_returns",
            "variance_units": "decimal_squared",
            "realized_measure": self.realized_measure,
            "intraday_realized_variance": False,
            "variance_family": REALIZED_GARCH_FAMILY,
            "vol": REALIZED_GARCH_FAMILY,
            "series_scope": self.series_scope,
            "min_obs": int(self.min_obs),
        }

    def predict(self, x: Any) -> NDArray[np.float64]:
        n = int(np.asarray(x).shape[0])
        return np.full(n, float(self.forecast(horizon=1)["sigma"][0]))

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="volatility",
            name=REALIZED_GARCH_FAMILY,
            version="v1",
            extra={
                "realized_measure": self.realized_measure,
                "intraday_realized_variance": False,
                "variance_family": REALIZED_GARCH_FAMILY,
                "series_scope": self.series_scope,
            },
        )
