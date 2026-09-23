"""Hansen-Huang-Shek log-linear Realized GARCH on daily Parkinson.

The realized measure is one-day Parkinson variance from daily OHLC, not
intraday realized variance and never a silent close-to-close r^2
substitute. This is a research-lab variance family, not a live-performance
or high-frequency RV claim.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from quant_fund.models.base import JoblibMixin, ModelMeta

REALIZED_GARCH_MEASURE = "parkinson_daily_ohlc"
REALIZED_GARCH_FAMILY = "realized_garch_log_linear"
_LN2 = float(np.log(2.0))
_LOG_H_BOUNDS = (-30.0, 15.0)
_GARCH_SCALE = 100.0
_GARCH_VARIANCE_FLOOR = 1e-16
_PERCENT_VARIANCE_SCALE = _GARCH_SCALE**2


def parkinson_daily_variance(
    high: NDArray[np.float64] | list[float],
    low: NDArray[np.float64] | list[float],
) -> NDArray[np.float64]:
    """One-day Parkinson variance from positive high/low levels.

    park = (log(H/L))^2 / (4 log 2). Undefined rows stay NaN rather than
    falling back to squared close-to-close returns.
    """
    highs = np.asarray(high, dtype=float).reshape(-1)
    lows = np.asarray(low, dtype=float).reshape(-1)
    if highs.shape != lows.shape:
        raise ValueError("Parkinson high and low must have the same length")
    out = np.full(highs.shape, np.nan, dtype=float)
    ok = (
        np.isfinite(highs)
        & np.isfinite(lows)
        & (highs > 0.0)
        & (lows > 0.0)
        & (highs >= lows)
    )
    if not np.any(ok):
        return out
    log_hl = np.log(highs[ok] / lows[ok])
    out[ok] = (log_hl * log_hl) / (4.0 * _LN2)
    return out


def _sigmoid(value: float, lo: float, hi: float) -> float:
    return float(lo + (hi - lo) / (1.0 + np.exp(-np.clip(value, -40.0, 40.0))))


class RealizedGARCHVol(JoblibMixin):
    """Causal log-linear Realized GARCH(1,1) with a Gaussian return law.

    Fitted to paired decimal ``returns`` and a strictly positive realized
    measure. Callers must pass ``realized_measure=`` explicitly; a forward
    variance label cannot be used as either series. Date-level overlays stamp
    ``series_scope=date_level_equal_weight_cross_section`` and do not replace
    ``vol_20``. ``forecast_asof`` / ``optimize_asof`` / paper-backtest
    ``check_order`` prefer the date-level Parkinson overlay when the artifact
    is present.
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
        if not isinstance(realized_measure, str) or not realized_measure.strip():
            raise ValueError("realized_measure must be a non-empty string")
        measure_key = realized_measure.strip()
        if measure_key != REALIZED_GARCH_MEASURE:
            raise ValueError(
                f"realized_measure must be {REALIZED_GARCH_MEASURE!r}; "
                "close-to-close squared returns and intraday RV are not accepted"
            )
        self.min_obs = min_obs
        self.mean = mean
        self.series_scope = series_scope.strip()
        self.realized_measure = measure_key
        self.intraday_realized_variance = False
        self.variance_family = REALIZED_GARCH_FAMILY
        self.result: dict[str, float] | None = None
        self.last_sigma: float = 0.01
        self.n_obs = 0
        self.fit_status = "unfitted"
        self.converged = False
        self.fallback_reason: str | None = None
        self.returns_scale = "decimal_returns_to_percent"
        self._returns_percent = np.empty(0, dtype=float)
        self._measure_percent = np.empty(0, dtype=float)
        self._last_h_percent = float("nan")
        self._last_x_percent = float("nan")

    def _fallback(self, returns_percent: NDArray[np.float64], reason: str) -> RealizedGARCHVol:
        self.result = None
        self.converged = False
        self.fit_status = "fallback"
        self.fallback_reason = reason
        if returns_percent.size > 1:
            sigma = float(np.std(returns_percent, ddof=1) / _GARCH_SCALE)
            self.last_sigma = sigma if np.isfinite(sigma) and sigma > 0.0 else 0.01
        else:
            self.last_sigma = 0.01
        self._last_h_percent = float(self.last_sigma * _GARCH_SCALE) ** 2
        self._last_x_percent = self._last_h_percent
        return self

    def fit_returns(
        self,
        returns: NDArray[np.float64],
        realized_measure: NDArray[np.float64],
    ) -> RealizedGARCHVol:
        """Fit directly on causal decimal returns and a paired realized measure."""
        values = np.asarray(returns, dtype=float).reshape(-1)
        return self.fit(
            np.empty((values.size, 0)),
            values,
            returns=values,
            realized_measure=realized_measure,
        )

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> RealizedGARCHVol:
        """Fit on causal returns plus ``realized_measure=``; ``y`` may be a label."""
        if "returns" not in kwargs:
            raise ValueError(
                "returns= must be supplied explicitly; y may be a forward label and "
                "cannot be used as a Realized GARCH likelihood input"
            )
        if "realized_measure" not in kwargs:
            raise ValueError(
                "realized_measure= must be supplied explicitly; close-to-close squared "
                "returns are not inferred"
            )
        try:
            values = np.asarray(kwargs.pop("returns"), dtype=float).reshape(-1)
            measure = np.asarray(kwargs.pop("realized_measure"), dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return self._fallback(np.empty(0), "invalid_returns")
        if values.shape != measure.shape:
            raise ValueError("returns and realized_measure must have the same length")
        finite = np.isfinite(values) & np.isfinite(measure) & (measure > 0.0)
        self.n_obs = int(finite.sum())
        returns_percent = values[finite] * _GARCH_SCALE
        measure_percent = measure[finite] * _PERCENT_VARIANCE_SCALE
        self._returns_percent = returns_percent.copy()
        self._measure_percent = measure_percent.copy()
        if self.n_obs < self.min_obs:
            return self._fallback(returns_percent, "insufficient_observations")
        if np.std(returns_percent) <= 0.0 or not np.isfinite(np.std(returns_percent)):
            return self._fallback(returns_percent, "zero_variance")
        if float(np.min(measure_percent)) <= 0.0:
            return self._fallback(returns_percent, "nonpositive_realized_measure")
        fitted = self._fit_qml(returns_percent, measure_percent)
        if fitted is None:
            return self._fallback(returns_percent, "nonconvergence")
        params, last_h, last_x = fitted
        if params["persistence"] >= 1.0:
            return self._fallback(returns_percent, "nonstationary_persistence")
        self.result = params
        self.converged = True
        self.fit_status = "fitted"
        self.fallback_reason = None
        self._last_h_percent = float(last_h)
        self._last_x_percent = float(last_x)
        self.last_sigma = float(np.sqrt(max(last_h, _GARCH_VARIANCE_FLOOR)) / _GARCH_SCALE)
        return self

    def _unpack(self, theta: NDArray[np.float64]) -> dict[str, float]:
        mu = 0.0 if self.mean == "Zero" else float(theta[0])
        omega = float(theta[1])
        beta = _sigmoid(float(theta[2]), 0.01, 0.98)
        gamma = float(np.exp(np.clip(theta[3], -8.0, 4.0)))
        xi = float(theta[4])
        phi = float(np.exp(np.clip(theta[5], -4.0, 2.0)))
        tau1 = float(theta[6])
        tau2 = float(theta[7])
        sigma_u = float(np.exp(np.clip(theta[8], -8.0, 4.0)))
        return {
            "mu": mu,
            "omega": omega,
            "beta": beta,
            "gamma": gamma,
            "xi": xi,
            "phi": phi,
            "tau1": tau1,
            "tau2": tau2,
            "sigma_u": sigma_u,
            "persistence": beta + gamma * phi,
        }

    def _joint_nll(
        self,
        theta: NDArray[np.float64],
        returns_percent: NDArray[np.float64],
        measure_percent: NDArray[np.float64],
    ) -> float:
        params = self._unpack(theta)
        if params["persistence"] >= 0.999 or params["sigma_u"] <= 0.0:
            return 1e12
        mu = params["mu"]
        log_x = np.log(np.clip(measure_percent, _GARCH_VARIANCE_FLOOR, None))
        uncond = (params["omega"] + params["gamma"] * params["xi"]) / (
            1.0 - params["persistence"]
        )
        log_h = float(np.clip(uncond, *_LOG_H_BOUNDS))
        nll = 0.0
        for ret, x_t, lx in zip(
            returns_percent.tolist(), measure_percent.tolist(), log_x.tolist(), strict=True
        ):
            h = float(np.exp(np.clip(log_h, *_LOG_H_BOUNDS)))
            if not np.isfinite(h) or h <= 0.0:
                return 1e12
            z = (float(ret) - mu) / np.sqrt(h)
            u = (
                lx
                - params["xi"]
                - params["phi"] * log_h
                - params["tau1"] * z
                - params["tau2"] * (z * z - 1.0)
            )
            nll += 0.5 * np.log(2.0 * np.pi) + 0.5 * np.log(h) + 0.5 * z * z
            nll += (
                0.5 * np.log(2.0 * np.pi)
                + np.log(params["sigma_u"])
                + 0.5 * (u / params["sigma_u"]) ** 2
            )
            log_h = params["omega"] + params["beta"] * log_h + params["gamma"] * np.log(
                max(float(x_t), _GARCH_VARIANCE_FLOOR)
            )
        if not np.isfinite(nll):
            return 1e12
        return float(nll / returns_percent.size)

    def _fit_qml(
        self,
        returns_percent: NDArray[np.float64],
        measure_percent: NDArray[np.float64],
    ) -> tuple[dict[str, float], float, float] | None:
        log_x = np.log(np.clip(measure_percent, _GARCH_VARIANCE_FLOOR, None))
        mu0 = 0.0 if self.mean == "Zero" else float(np.mean(returns_percent))
        log_h0 = float(np.mean(log_x))
        beta0, gamma0, phi0 = 0.55, 0.35, 1.0
        omega0 = (1.0 - beta0 - gamma0 * phi0) * log_h0
        xi0 = float(np.mean(log_x) - phi0 * log_h0)
        x0 = np.array(
            [
                mu0,
                omega0,
                0.4,  # logit-ish start for beta ~ 0.55
                np.log(gamma0),
                xi0,
                np.log(phi0),
                0.0,
                0.1,
                np.log(max(float(np.std(log_x, ddof=1)), 0.05)),
            ],
            dtype=float,
        )

        def objective(theta: NDArray[np.float64]) -> float:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                return self._joint_nll(theta, returns_percent, measure_percent)

        try:
            result = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": 400})
            if (not bool(result.success)) or (not np.isfinite(result.fun)):
                result = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": 800})
        except (ValueError, FloatingPointError, TypeError):
            return None
        if not bool(result.success) or not np.isfinite(result.fun):
            return None
        params = self._unpack(np.asarray(result.x, dtype=float))
        path = self._filter_path(params, returns_percent, measure_percent)
        if path is None:
            return None
        last_h, last_x = path
        params["mu"] = float(params["mu"])
        return params, last_h, last_x

    def _filter_path(
        self,
        params: dict[str, float],
        returns_percent: NDArray[np.float64],
        measure_percent: NDArray[np.float64],
    ) -> tuple[float, float] | None:
        if returns_percent.size != measure_percent.size:
            return None
        uncond = (params["omega"] + params["gamma"] * params["xi"]) / (
            1.0 - params["persistence"]
        )
        log_h = float(np.clip(uncond, *_LOG_H_BOUNDS))
        last_h = float(np.exp(log_h))
        last_x = float(measure_percent[0])
        for x_t in measure_percent.tolist():
            h = float(np.exp(np.clip(log_h, *_LOG_H_BOUNDS)))
            if not np.isfinite(h) or h <= 0.0:
                return None
            last_h = h
            last_x = float(x_t)
            log_h = params["omega"] + params["beta"] * log_h + params["gamma"] * np.log(
                max(float(x_t), _GARCH_VARIANCE_FLOOR)
            )
        if not np.isfinite(last_h) or last_h <= 0.0 or not np.isfinite(last_x) or last_x <= 0.0:
            return None
        return last_h, last_x

    def _mean_decimal(self) -> float:
        if self.result is None or self.mean == "Zero":
            return 0.0
        return float(self.result["mu"]) / _GARCH_SCALE

    def forecast(
        self,
        horizon: int = 1,
        *,
        method: str = "analytic",
        simulations: int = 1000,
        quantiles: tuple[float, ...] | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Origin-indexed decimal variance path. Multi-step uses E[log h] plugin."""
        del simulations, seed
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if method not in {"analytic", "simulation", "bootstrap"}:
            raise ValueError("method must be analytic, simulation, or bootstrap")
        if self.result is None:
            variance = np.full(horizon, max(self.last_sigma**2, _GARCH_VARIANCE_FLOOR))
            out: dict[str, Any] = {
                "variance": variance,
                "sigma": np.sqrt(variance),
                "cumulative_variance": np.cumsum(variance),
                "mean": self._mean_decimal(),
                "distribution": "normal",
                "requested_distribution": "normal",
                "horizon": horizon,
                "fit_status": self.fit_status,
                "realized_measure": self.realized_measure,
                "intraday_realized_variance": False,
                "multi_step_method": "fallback_constant",
            }
            if quantiles is not None:
                levels = np.asarray(quantiles, dtype=float)
                if levels.ndim != 1 or np.any((levels <= 0.0) | (levels >= 1.0)):
                    raise ValueError("quantiles must lie strictly between 0 and 1")
                from scipy.stats import norm

                out["quantiles"] = (
                    self.last_sigma * norm.ppf(levels)[None, :] + self._mean_decimal()
                )
            return out
        params = self.result
        log_h = float(np.log(max(self._last_h_percent, _GARCH_VARIANCE_FLOOR)))
        log_x = float(np.log(max(self._last_x_percent, _GARCH_VARIANCE_FLOOR)))
        variance_percent = np.empty(horizon, dtype=float)
        for step in range(horizon):
            log_h = params["omega"] + params["beta"] * log_h + params["gamma"] * log_x
            h = float(np.exp(np.clip(log_h, *_LOG_H_BOUNDS)))
            variance_percent[step] = max(h, _GARCH_VARIANCE_FLOOR)
            # Future realized measures are not observed; plugin E[log x]=ξ+φ log h.
            log_x = params["xi"] + params["phi"] * log_h
        variance = variance_percent / _PERCENT_VARIANCE_SCALE
        variance = np.maximum(variance, _GARCH_VARIANCE_FLOOR)
        out = {
            "variance": variance,
            "sigma": np.sqrt(variance),
            "cumulative_variance": np.cumsum(variance),
            "mean": self._mean_decimal(),
            "distribution": "normal",
            "requested_distribution": "normal",
            "horizon": horizon,
            "fit_status": self.fit_status,
            "realized_measure": self.realized_measure,
            "intraday_realized_variance": False,
            "multi_step_method": "expected_log_variance_plugin",
        }
        if quantiles is not None:
            levels = np.asarray(quantiles, dtype=float)
            if levels.ndim != 1 or np.any((levels <= 0.0) | (levels >= 1.0)):
                raise ValueError("quantiles must lie strictly between 0 and 1")
            from scipy.stats import norm

            out["quantiles"] = (
                np.sqrt(variance)[:, None] * norm.ppf(levels)[None, :] + self._mean_decimal()
            )
        return out

    def pit(
        self, returns: NDArray[np.float64], sigma: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        values = np.asarray(returns, dtype=float).reshape(-1)
        scales = (
            np.full(values.size, self.last_sigma)
            if sigma is None
            else np.asarray(sigma, dtype=float).reshape(-1)
        )
        if values.size != scales.size:
            raise ValueError("returns and sigma must have the same length")
        valid = np.isfinite(values) & np.isfinite(scales) & (scales > 0.0)
        out = np.full(values.size, np.nan)
        from scipy.stats import norm

        out[valid] = norm.cdf((values[valid] - self._mean_decimal()) / scales[valid])
        return np.clip(out, 0.0, 1.0)

    def log_density(
        self, returns: NDArray[np.float64], sigma: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        values = np.asarray(returns, dtype=float).reshape(-1)
        scales = (
            np.full(values.size, self.last_sigma)
            if sigma is None
            else np.asarray(sigma, dtype=float).reshape(-1)
        )
        if values.size != scales.size:
            raise ValueError("returns and sigma must have the same length")
        valid = np.isfinite(values) & np.isfinite(scales) & (scales > 0.0)
        out = np.full(values.size, np.nan)
        if not np.any(valid):
            return out
        z = (values[valid] - self._mean_decimal()) / scales[valid]
        out[valid] = -0.5 * np.log(2.0 * np.pi) - np.log(scales[valid]) - 0.5 * z * z
        return out

    def diagnostics(self) -> dict[str, Any]:
        return {
            "n_obs": int(self.n_obs),
            "fit_status": self.fit_status,
            "converged": bool(self.converged),
            "fallback_reason": self.fallback_reason,
            "returns_scale": self.returns_scale,
            "variance_units": "decimal_squared",
            "vol": "realized_garch",
            "dist": "normal",
            "series_scope": self.series_scope,
            "realized_measure": self.realized_measure,
            "intraday_realized_variance": False,
            "variance_family": self.variance_family,
            "persistence": None if self.result is None else float(self.result["persistence"]),
        }

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        n = int(np.asarray(x).shape[0])
        return np.full(n, float(self.forecast(horizon=1)["sigma"][0]))

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="volatility",
            name="realized_garch_parkinson",
            version="v1",
            extra={
                "realized_measure": self.realized_measure,
                "intraday_realized_variance": False,
                "variance_family": self.variance_family,
                "series_scope": self.series_scope,
            },
        )
