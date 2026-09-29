"""Volatility training and name-level GARCH walk-forward.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.metrics.scoring import (
    GARCH_ONE_STEP_CRPS_TAUS,
    crps_from_quantiles,
    crps_gaussian,
    name_level_one_step_density_summary,
    name_level_qlike,
    one_step_density_summary,
    overlap_aware_qlike,
    qlike,
)
from quant_fund.models.base import load_joblib_artifact, save_joblib_artifact
from quant_fund.models.realized_garch import REALIZED_GARCH_MEASURE, RealizedGARCHVol
from quant_fund.models.volatility import (
    GARCH_DATE_LEVEL_SCOPE,
    GARCH_SECURITY_LEVEL_SCOPE,
    EWMAVol,
    GARCHVol,
    HARVol,
    RollingVol,
    TreeVol,
)
from quant_fund.pipeline.dataset import design_matrix, panel
from quant_fund.registry.mlflow_store import configure_tracking, log_run
from quant_fund.schemas.errors import PointInTimeError
from quant_fund.utils.seeds import set_global_seed

from .garch_data import (
    _garch_name_return_history,
    _garch_return_history,
    _realized_garch_history,
    _require_garch_security_keys,
)
from .splits import _aligned_label_end_times, _label_horizon, _require_model, _walk_forward_splits


def _garch_oos_predictions(
    make_model: Callable[[], Any],
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    test_mask: np.ndarray,
    *,
    label_horizon: int,
    return_dates: np.ndarray,
    return_values: np.ndarray,
    return_frame: Any | None = None,
) -> tuple[np.ndarray, list[str], dict[Any, dict[str, Any]]]:
    """Forecast each test row from the last information available at its origin.

    GARCH is a stateful time-series model, so one forecast from the fold boundary
    cannot be repeated across a multi-date test window.  Refit on the expanding
    history strictly before each test date; this is causal and lets the return
    likelihood absorb information that was available before that origin.
    ``return_dates``/``return_values`` come from the full feature panel, not the
    label-filtered design matrix, so structurally unavailable forward labels cannot
    remove otherwise usable historical returns.

    When ``return_frame`` carries ``available_time``, each origin refits on the
    PIT-observable equal-weight series rather than a pre-aggregated path that
    already includes unpublished restatements. Density targets still use the
    date-level ``ret_1`` outcome at the origin (evaluation vintage).

    One-step density records score the date-level ``ret_1`` at the origin against
    the predictive law fitted on strictly prior returns.  That object is not the
    *h*-bar realized-variance QLIKE target and is never a multi-step Gaussian
    approximation to ``cumulative_variance``.
    """
    date_values = np.asarray(dates)
    mask = np.asarray(test_mask, dtype=bool)
    history_dates = np.asarray(return_dates)
    history_values = np.asarray(return_values, dtype=float).reshape(-1)
    if date_values.ndim != 1 or mask.shape != date_values.shape:
        raise ValueError("dates and test_mask must be one-dimensional and aligned")
    if history_dates.ndim != 1 or history_dates.shape != history_values.shape:
        raise ValueError("return_dates and return_values must be one-dimensional and aligned")
    if history_values.size == 0:
        raise ValueError("GARCH return history must not be empty")
    if not np.all(np.isfinite(history_values)):
        raise ValueError("GARCH return history must be finite")
    if label_horizon < 1:
        raise ValueError("label_horizon must be positive")
    test_indices = np.flatnonzero(mask)
    if test_indices.size == 0:
        return np.empty(0, dtype=float), [], {}

    history_lookup = {
        date: float(value)
        for date, value in zip(history_dates.tolist(), history_values.tolist(), strict=True)
    }
    pit_frame = (
        return_frame
        if return_frame is not None and "available_time" in getattr(return_frame, "columns", ())
        else None
    )
    by_date: dict[Any, float] = {}
    density_by_date: dict[Any, dict[str, Any]] = {}
    statuses: list[str] = []
    for test_date in sorted(set(date_values[test_indices].tolist())):
        origin_mask = date_values < test_date
        if pit_frame is not None:
            _hist_dates, historical_returns = _garch_return_history(pit_frame, asof=test_date)
            if historical_returns.size == 0:
                raise ValueError("GARCH test origin has no strictly prior return observations")
        else:
            return_origin_mask = history_dates < test_date
            if not np.any(return_origin_mask):
                raise ValueError("GARCH test origin has no strictly prior return observations")
            historical_returns = history_values[return_origin_mask]
            if historical_returns.size == 0:
                raise ValueError("GARCH test origin has no causal returns")
        model = make_model()
        model.fit(x[origin_mask], y[origin_mask], returns=historical_returns)
        forecast = model.forecast(horizon=label_horizon)
        cumulative = np.asarray(forecast["cumulative_variance"], dtype=float).reshape(-1)
        if cumulative.size < label_horizon or not np.isfinite(cumulative[label_horizon - 1]):
            raise ValueError("GARCH produced an invalid out-of-sample forecast")
        by_date[test_date] = float(cumulative[label_horizon - 1])
        statuses.append(str(getattr(model, "fit_status", "unknown")))
        if hasattr(model, "log_density") and hasattr(model, "pit"):
            if test_date not in history_lookup:
                raise ValueError("GARCH one-step density target missing origin ret_1")
            density_by_date[test_date] = _garch_origin_density_record(
                model, history_lookup[test_date], forecast
            )

    predictions = np.asarray(
        [by_date[date_values.tolist()[int(index)]] for index in test_indices], dtype=float
    )
    return predictions, statuses, density_by_date


def _realized_garch_oos_predictions(
    make_model: Callable[[], Any],
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    test_mask: np.ndarray,
    *,
    label_horizon: int,
    return_frame: Any,
) -> tuple[np.ndarray, list[str], dict[Any, dict[str, Any]]]:
    """Walk-forward Realized GARCH using strictly prior Parkinson/return pairs.

    The realized measure is one-day Parkinson from daily OHLC. Origin-bar
    OHLC cannot enter the fit; density still scores the origin's date-level
    ``ret_1`` from the same paired cross-section (evaluation vintage).
    """
    date_values = np.asarray(dates)
    mask = np.asarray(test_mask, dtype=bool)
    if date_values.ndim != 1 or mask.shape != date_values.shape:
        raise ValueError("dates and test_mask must be one-dimensional and aligned")
    if return_frame is None:
        raise ValueError("Realized GARCH walk-forward requires a return frame")
    if label_horizon < 1:
        raise ValueError("label_horizon must be positive")
    test_indices = np.flatnonzero(mask)
    if test_indices.size == 0:
        return np.empty(0, dtype=float), [], {}
    eval_dates, eval_returns, _eval_measures = _realized_garch_history(return_frame)
    history_lookup = {
        date: float(value)
        for date, value in zip(eval_dates.tolist(), eval_returns.tolist(), strict=True)
    }
    by_date: dict[Any, float] = {}
    density_by_date: dict[Any, dict[str, Any]] = {}
    statuses: list[str] = []
    for test_date in sorted(set(date_values[test_indices].tolist())):
        origin_mask = date_values < test_date
        _historical_dates, historical_returns, historical_measures = _realized_garch_history(
            return_frame, asof=test_date
        )
        if historical_returns.size == 0 or historical_measures.size == 0:
            raise ValueError("Realized GARCH test origin has no strictly prior Parkinson pairs")
        model = make_model()
        model.fit(
            x[origin_mask],
            y[origin_mask],
            returns=historical_returns,
            realized_measure=historical_measures,
        )
        forecast = model.forecast(horizon=label_horizon)
        cumulative = np.asarray(forecast["cumulative_variance"], dtype=float).reshape(-1)
        if cumulative.size < label_horizon or not np.isfinite(cumulative[label_horizon - 1]):
            raise ValueError("Realized GARCH produced an invalid out-of-sample forecast")
        by_date[test_date] = float(cumulative[label_horizon - 1])
        statuses.append(str(getattr(model, "fit_status", "unknown")))
        if test_date not in history_lookup:
            raise ValueError("Realized GARCH one-step density target missing origin ret_1")
        density_by_date[test_date] = _garch_origin_density_record(
            model, history_lookup[test_date], forecast
        )
    predictions = np.asarray(
        [by_date[date_values.tolist()[int(index)]] for index in test_indices], dtype=float
    )
    return predictions, statuses, density_by_date


def _garch_name_origin_return_lookup(frame: Any) -> dict[tuple[str, Any], float]:
    """Evaluation-vintage per-name ``ret_1`` at each origin (not PIT-filtered)."""
    _require_garch_security_keys(frame)
    columns = set(frame.columns)
    if not {"event_time", "security_id", "ret_1"}.issubset(columns):
        raise ValueError("per-security GARCH density requires event_time, security_id, and ret_1")
    history = frame.select(["event_time", "security_id", "ret_1"]).drop_nulls()
    if history.is_empty():
        return {}
    dates = np.asarray(history["event_time"].to_numpy())
    ids = history["security_id"].to_list()
    values = np.asarray(history["ret_1"].to_numpy(), dtype=float)
    lookup: dict[tuple[str, Any], float] = {}
    for sid_raw, date, value in zip(ids, dates.tolist(), values.tolist(), strict=True):
        if isinstance(sid_raw, bool) or sid_raw is None or not isinstance(sid_raw, str):
            raise ValueError("per-security GARCH security_ids must be strings")
        sid = sid_raw.strip()
        if not sid:
            raise ValueError("per-security GARCH security_ids must be non-empty")
        if not np.isfinite(value):
            continue
        key = (sid, date)
        if key in lookup:
            raise PointInTimeError(
                "per-security GARCH origin returns contain duplicate security_id/event_time"
            )
        lookup[key] = float(value)
    return lookup


def _garch_name_row_id(raw: object) -> str:
    if isinstance(raw, bool) or raw is None or not isinstance(raw, str):
        raise ValueError("per-security GARCH security_ids must be strings")
    sid = raw.strip()
    if not sid:
        raise ValueError("per-security GARCH security_ids must be non-empty")
    return sid


def _garch_vol_spec(model: Any) -> dict[str, Any] | None:
    """Spec dict for a stock ``GARCHVol``. Subclasses and stand-ins stay serial."""
    if type(model) is not GARCHVol:
        return None
    return {
        "p": int(model.p),
        "q": int(model.q),
        "dist": str(model.dist),
        "vol": str(model.vol),
        "min_obs": int(model.min_obs),
        "mean": str(model.mean),
        "power": float(model.power),
        "series_scope": str(model.series_scope),
    }


def _garch_origin_task(
    item: tuple[dict[str, Any], np.ndarray, int, float | None], _seed: int
) -> tuple[float, str, dict[str, Any]]:
    """Fit one name-origin and forecast it. ``_seed`` is unused.

    ``forecast`` keeps the historical default seed (``None`` → ``RandomState(0)``
    for simulated multi-step EGARCH/APARCH). An index-dependent seed would
    change those paths relative to the serial loop.
    """
    spec, historical_returns, label_horizon, origin_return = item
    model = GARCHVol(**spec)
    model.fit(
        np.zeros((0, 1), dtype=float),
        np.zeros(0, dtype=float),
        returns=historical_returns,
    )
    forecast = model.forecast(horizon=int(label_horizon))
    cumulative = np.asarray(forecast["cumulative_variance"], dtype=float).reshape(-1)
    if cumulative.size < int(label_horizon) or not np.isfinite(cumulative[int(label_horizon) - 1]):
        raise ValueError("GARCH produced an invalid out-of-sample forecast")
    if origin_return is None:
        raise ValueError("per-security GARCH one-step density target missing origin ret_1")
    density = _garch_origin_density_record(model, float(origin_return), forecast)
    return (
        float(cumulative[int(label_horizon) - 1]),
        str(getattr(model, "fit_status", "unknown")),
        density,
    )


def _garch_name_oos_predictions(
    make_model: Callable[[], Any],
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    ids: np.ndarray,
    test_mask: np.ndarray,
    *,
    label_horizon: int,
    return_frame: Any,
) -> tuple[np.ndarray, list[str], dict[tuple[str, Any], dict[str, Any]]]:
    """Forecast each test name from that name's strictly prior ``ret_1``.

    Date-level ``_garch_oos_predictions`` pools names into an equal-weight
    market series. This path never does that: each ``security_id`` is a
    separate univariate GARCH. QLIKE targets remain the row's
    ``future_realized_var_h``; one-step density targets that name's origin
    ``ret_1`` (evaluation vintage), never the cross-section mean.
    ``available_time`` uses the same as-of contract as name-level as-of
    forecasts. Duplicate ``(security_id, event_time)`` test keys fail closed.
    """
    date_values = np.asarray(dates)
    id_values = np.asarray(ids)
    mask = np.asarray(test_mask, dtype=bool)
    if date_values.ndim != 1 or mask.shape != date_values.shape:
        raise ValueError("dates and test_mask must be one-dimensional and aligned")
    if id_values.shape != date_values.shape:
        raise ValueError("ids and dates must be one-dimensional and aligned")
    if x.shape[0] != date_values.shape[0] or y.shape[0] != date_values.shape[0]:
        raise ValueError("x, y, dates, and ids must be aligned")
    if label_horizon < 1:
        raise ValueError("label_horizon must be positive")
    if return_frame is None:
        raise ValueError("per-security GARCH walk-forward requires a return frame")
    test_indices = np.flatnonzero(mask)
    if test_indices.size == 0:
        return np.empty(0, dtype=float), [], {}

    origin_returns = _garch_name_origin_return_lookup(return_frame)
    by_key: dict[tuple[str, Any], float] = {}
    density_by_key: dict[tuple[str, Any], dict[str, Any]] = {}
    statuses: list[str] = []
    prepared: list[tuple[tuple[str, Any], np.ndarray, float | None]] = []
    for index in test_indices.tolist():
        sid = _garch_name_row_id(id_values.tolist()[int(index)])
        test_date = date_values.tolist()[int(index)]
        key = (sid, test_date)
        if key in by_key:
            raise ValueError("per-security GARCH walk-forward has duplicate security_id/event_time")
        # Reserve the key before the fit so a later duplicate still fails in input order.
        by_key[key] = float("nan")
        _hist_dates, historical_returns = _garch_name_return_history(
            return_frame, sid, asof=test_date
        )
        if historical_returns.size == 0:
            raise ValueError(
                f"per-security GARCH test origin has no strictly prior returns for {sid!r}"
            )
        origin_return = origin_returns.get(key)
        prepared.append(
            (
                key,
                np.ascontiguousarray(historical_returns, dtype=np.float64),
                None if origin_return is None else float(origin_return),
            )
        )

    spec = _garch_vol_spec(make_model())
    if spec is None:
        # Custom factories are not assumed picklable. Keep the original serial fits.
        dummy_x = np.zeros((0, 1), dtype=float)
        dummy_y = np.zeros(0, dtype=float)
        for key, historical_returns, origin_return in prepared:
            model = make_model()
            model.fit(dummy_x, dummy_y, returns=historical_returns)
            forecast = model.forecast(horizon=label_horizon)
            cumulative = np.asarray(forecast["cumulative_variance"], dtype=float).reshape(-1)
            if cumulative.size < label_horizon or not np.isfinite(cumulative[label_horizon - 1]):
                raise ValueError("GARCH produced an invalid out-of-sample forecast")
            by_key[key] = float(cumulative[label_horizon - 1])
            statuses.append(str(getattr(model, "fit_status", "unknown")))
            if hasattr(model, "log_density") and hasattr(model, "pit"):
                if origin_return is None:
                    raise ValueError(
                        "per-security GARCH one-step density target missing origin ret_1"
                    )
                density_by_key[key] = _garch_origin_density_record(model, origin_return, forecast)
    else:
        # Each origin is an independent fit. A process pool was slower than
        # this in-process loop on sub-second batches (pool startup dominated
        # ~10ms GARCH fits), so the sweep stays here. ``_garch_origin_task``
        # is the same function the pool tests call. The seed is unused.
        fitted = [
            _garch_origin_task((spec, hist, int(label_horizon), origin), 0)
            for _key, hist, origin in prepared
        ]
        for (key, _hist, _origin), (prediction, status, density) in zip(
            prepared, fitted, strict=True
        ):
            by_key[key] = prediction
            statuses.append(status)
            density_by_key[key] = density

    predictions = np.asarray(
        [
            by_key[
                (
                    _garch_name_row_id(id_values.tolist()[int(index)]),
                    date_values.tolist()[int(index)],
                )
            ]
            for index in test_indices.tolist()
        ],
        dtype=float,
    )
    return predictions, statuses, density_by_key


def _garch_one_step_sigma(forecast: dict[str, Any]) -> float:
    sigma_path = forecast.get("sigma")
    var_path = forecast.get("variance")
    if sigma_path is not None:
        sigma = float(np.asarray(sigma_path, dtype=float).reshape(-1)[0])
    elif var_path is not None:
        variance = float(np.asarray(var_path, dtype=float).reshape(-1)[0])
        sigma = (
            float(np.sqrt(variance)) if np.isfinite(variance) and variance > 0.0 else float("nan")
        )
    else:
        raise ValueError("GARCH one-step density requires forecast sigma or variance")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("GARCH produced an invalid one-step sigma")
    return sigma


def _garch_origin_density_record(
    model: Any, realized: float, forecast: dict[str, Any]
) -> dict[str, Any]:
    """Score the one-step predictive law against date-level ``ret_1`` at the origin."""
    if not np.isfinite(realized):
        raise ValueError("GARCH one-step density target must be finite")
    sigma = _garch_one_step_sigma(forecast)
    y = np.array([float(realized)], dtype=float)
    sig = np.array([sigma], dtype=float)
    log_s = np.asarray(model.log_density(y, sig), dtype=float).reshape(-1)
    pit = np.asarray(model.pit(y, sig), dtype=float).reshape(-1)
    if log_s.size != 1 or pit.size != 1:
        raise ValueError("GARCH one-step density produced a malformed score")
    dist = str(forecast.get("distribution", "normal"))
    try:
        mu = float(forecast["mean"])
    except (KeyError, TypeError, ValueError):
        mu = float(model._mean_decimal()) if hasattr(model, "_mean_decimal") else 0.0
    if dist == "normal":
        crps = float(crps_gaussian(y, np.array([mu], dtype=float), sig)[0])
        crps_method = "gaussian_closed"
    else:
        taus = np.asarray(GARCH_ONE_STEP_CRPS_TAUS, dtype=float)
        quantile_forecast = model.forecast(horizon=1, quantiles=tuple(taus.tolist()))
        quantiles = np.asarray(quantile_forecast["quantiles"], dtype=float)
        crps = float(crps_from_quantiles(y, quantiles, taus))
        crps_method = "quantile_riemann"
    return {
        "y": float(realized),
        "mu": mu,
        "sigma": sigma,
        "distribution": dist,
        "requested_distribution": str(forecast.get("requested_distribution", dist)),
        "log_score": float(log_s[0]),
        "crps": crps,
        "crps_method": crps_method,
        "pit": float(pit[0]),
        "fit_status": str(forecast.get("fit_status", "unknown")),
        "density_horizon": 1,
    }


def _empty_garch_density_metrics() -> dict[str, Any]:
    return {
        "log_score_one_step": float("nan"),
        "ignorance_one_step": float("nan"),
        "crps_one_step": float("nan"),
        "pit_ks_one_step": float("nan"),
        "pit_ks_p_one_step": float("nan"),
        "n_density_origins": 0,
        "n_density_origins_qlike_stride": 0,
        "log_score_one_step_qlike_origins": float("nan"),
        "crps_one_step_qlike_origins": float("nan"),
        "density_horizon": 1,
        "density_target": "date_level_ret_1",
    }


def train_volatility(config: AppConfig, model_name: str = "ewma") -> dict[str, Any]:
    _require_model(
        model_name,
        {"rolling", "ewma", "garch", "realized_garch", "har", "xgboost", "lightgbm"},
        "volatility",
    )
    set_global_seed(config.train.random_seed)
    label = config.train.volatility_target
    df = panel(config, label=label)
    x, y, dates, feats, _ = design_matrix(
        df, label, ["vol_20", "vol_ewma", "vol_parkinson", "ret_1", "vol_of_vol"]
    )
    label_end_times = _aligned_label_end_times(df, label, feats)
    return_dates, return_values = (
        _garch_return_history(df) if model_name == "garch" else (np.empty(0), np.empty(0))
    )
    _realized_dates, realized_returns, realized_measures = (
        _realized_garch_history(df)
        if model_name == "realized_garch"
        else (np.empty(0), np.empty(0), np.empty(0))
    )

    def make_model() -> Any:
        catalog: dict[str, Callable[[], Any]] = {
            "rolling": RollingVol,
            "ewma": lambda: EWMAVol(config.features.ewma_lambda),
            "garch": lambda: GARCHVol(
                p=config.train.garch_p,
                q=config.train.garch_q,
                dist=config.train.garch_dist.value,
                vol=config.train.garch_vol.value,
                series_scope=GARCH_DATE_LEVEL_SCOPE,
            ),
            "realized_garch": lambda: RealizedGARCHVol(
                series_scope=GARCH_DATE_LEVEL_SCOPE,
                realized_measure=REALIZED_GARCH_MEASURE,
            ),
            "har": lambda: HARVol(config.train.har_log),
            "xgboost": lambda: TreeVol("xgboost", config.train.random_seed),
            "lightgbm": lambda: TreeVol("lightgbm", config.train.random_seed),
        }
        if model_name not in catalog:
            raise ValueError(f"unknown volatility model {model_name!r}")
        return catalog[model_name]()

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    eval_dates: list[np.ndarray] = []
    fold_status: list[str] = []
    density_by_date: dict[Any, dict[str, Any]] = {}
    label_horizon = _label_horizon(label)

    for train_mask, test_mask in _walk_forward_splits(
        dates,
        config,
        horizon_bars=label_horizon,
        label_end_times=label_end_times,
    ):
        if not train_mask.any() or not test_mask.any():
            continue
        if model_name == "garch":
            fold_predictions, statuses, fold_density = _garch_oos_predictions(
                make_model,
                x,
                y,
                dates,
                test_mask,
                label_horizon=label_horizon,
                return_dates=return_dates,
                return_values=return_values,
                return_frame=df,
            )
            predictions.append(fold_predictions)
            fold_status.extend(statuses)
            for origin, record in fold_density.items():
                if origin in density_by_date:
                    raise ValueError("GARCH one-step density origin repeated across folds")
                density_by_date[origin] = record
        elif model_name == "realized_garch":
            fold_predictions, statuses, fold_density = _realized_garch_oos_predictions(
                make_model,
                x,
                y,
                dates,
                test_mask,
                label_horizon=label_horizon,
                return_frame=df,
            )
            predictions.append(fold_predictions)
            fold_status.extend(statuses)
            for origin, record in fold_density.items():
                if origin in density_by_date:
                    raise ValueError("Realized GARCH one-step density origin repeated across folds")
                density_by_date[origin] = record
        else:
            fold_model = make_model()
            fold_model.fit(x[train_mask], y[train_mask])
            # Rolling/EWMA models expose sigma forecasts.  Select the feature
            # they actually represent, then convert sigma to variance because
            # qlike() is defined on the forward realized-variance target.
            if model_name == "rolling":
                sigma_features = x[:, 0:1]
            elif model_name == "ewma":
                sigma_features = x[:, 1:2]
            else:
                sigma_features = x
            sigma = np.asarray(fold_model.predict(sigma_features[test_mask]), dtype=float)
            predictions.append(np.square(sigma))
        targets.append(y[test_mask])
        eval_dates.append(np.asarray(dates[test_mask]))
    if not predictions:
        raise ValueError("walk-forward training produced no trainable/evaluable fold")
    pred = np.clip(np.concatenate(predictions), config.train.qlike_floor, None)
    yy = np.concatenate(targets)
    if model_name in {"garch", "realized_garch"}:
        session_index = {date: i for i, date in enumerate(sorted(set(np.asarray(dates).tolist())))}
        scored = overlap_aware_qlike(
            np.concatenate(eval_dates),
            pred,
            yy,
            horizon_bars=label_horizon,
            floor=config.train.qlike_floor,
            session_index=session_index,
        )
        metrics: dict[str, Any] = {
            "qlike": float(scored["qlike"]),
            "n_oos_rows": float(yy.size),
            "qlike_overlapping_dates": float(scored["qlike_overlapping"]),
            "n_origins_nonoverlapping": int(scored["n_origins_nonoverlapping"]),
            "n_origins_overlapping": int(scored["n_origins_overlapping"]),
            "origin_stride": int(scored["horizon_bars"]),
            "scoring_scope": GARCH_DATE_LEVEL_SCOPE,
        }
        if model_name == "realized_garch":
            metrics["realized_measure"] = REALIZED_GARCH_MEASURE
            metrics["intraday_realized_variance"] = False
        if density_by_date:
            origin_dates = np.asarray(sorted(density_by_date), dtype=object)
            density_metrics = one_step_density_summary(
                origin_dates,
                np.asarray(
                    [density_by_date[date]["log_score"] for date in origin_dates.tolist()],
                    dtype=float,
                ),
                np.asarray(
                    [density_by_date[date]["crps"] for date in origin_dates.tolist()],
                    dtype=float,
                ),
                np.asarray(
                    [density_by_date[date]["pit"] for date in origin_dates.tolist()],
                    dtype=float,
                ),
                horizon_bars=label_horizon,
                session_index=session_index,
            )
            density_metrics.pop("scoring_scope", None)
            density_metrics.pop("horizon_bars", None)
            crps_methods = {
                str(density_by_date[date]["crps_method"]) for date in origin_dates.tolist()
            }
            density_metrics["crps_method_one_step"] = (
                crps_methods.pop() if len(crps_methods) == 1 else "mixed"
            )
            metrics.update(density_metrics)
        else:
            metrics.update(_empty_garch_density_metrics())
        if fold_status:
            metrics["fallback_rate"] = float(np.mean(np.asarray(fold_status) != "fitted"))
        if model_name == "realized_garch":
            model = make_model().fit(
                x, y, returns=realized_returns, realized_measure=realized_measures
            )
        else:
            model = make_model().fit(x, y, returns=return_values)
    else:
        metrics = {"qlike": qlike(yy, pred, config.train.qlike_floor)}
        model = make_model().fit(x, y)
    garch_params = {
        "garch_p": config.train.garch_p,
        "garch_q": config.train.garch_q,
        "garch_dist": config.train.garch_dist.value,
        "garch_vol": config.train.garch_vol.value,
        "target": label,
        "target_contract": "forward_realized_variance_evaluation_only",
        "fit_input": "ret_1_decimal_returns",
        "forecast_horizon": label_horizon,
        "variance_units": "decimal_squared",
        "series_scope": GARCH_DATE_LEVEL_SCOPE,
        "oos_scoring": "date_level_nonoverlapping_qlike+one_step_density",
        "origin_stride": label_horizon,
    }
    if model_name == "realized_garch":
        garch_params.update(
            {
                "fit_input": "ret_1_and_parkinson_daily_ohlc",
                "realized_measure": REALIZED_GARCH_MEASURE,
                "intraday_realized_variance": False,
                "garch_vol": "realized_garch",
            }
        )
    configure_tracking()
    run_id = log_run(
        family="volatility",
        name=model_name,
        params=garch_params if model_name in {"garch", "realized_garch"} else {},
        metrics=metrics,
        tags={"data": config.data.source},
    )
    path = Path(config.data.root) / "metadata" / f"vol_{model_name}.joblib"
    model.save(path)
    result = {"metrics": metrics, "run_id": run_id, "path": str(path)}
    if model_name in {"garch", "realized_garch"}:
        diagnostics: dict[str, Any] = {
            "fit_status": model.fit_status,
            "converged": model.converged,
            "fallback_reason": model.fallback_reason,
            "n_obs": model.n_obs,
            "returns_scale": model.returns_scale,
            "series_scope": GARCH_DATE_LEVEL_SCOPE,
            "oos_scoring": "date_level_nonoverlapping_qlike+one_step_density",
            "n_origins_nonoverlapping": int(metrics["n_origins_nonoverlapping"]),
            "n_origins_overlapping": int(metrics["n_origins_overlapping"]),
            "origin_stride": label_horizon,
            "n_density_origins": int(metrics["n_density_origins"]),
            "density_target": "date_level_ret_1",
            "density_horizon": 1,
        }
        if model_name == "realized_garch":
            diagnostics["realized_measure"] = REALIZED_GARCH_MEASURE
            diagnostics["intraday_realized_variance"] = False
        result["diagnostics"] = diagnostics
    return result


def train_volatility_auto(config: AppConfig) -> dict[str, Any]:
    """Select volatility by finite causal walk-forward QLIKE."""
    candidates = ["rolling", "ewma", "har", "xgboost", "lightgbm"]
    results = [train_volatility(config, name) for name in candidates]
    eligible = [
        r
        for r in results
        if np.isfinite(float(r["metrics"].get("qlike", np.nan)))
        and float(r["metrics"].get("n_oos_rows", 0.0)) >= config.train.auto_min_oos_rows
    ]
    if not eligible:
        raise ValueError("automatic volatility selection produced no finite candidate metric")
    selected = min(eligible, key=lambda r: float(r["metrics"]["qlike"]))
    payload = load_joblib_artifact(Path(str(selected["path"])))
    auto_path = Path(config.data.root) / "metadata" / "vol_auto.joblib"
    save_joblib_artifact(payload, auto_path)
    selected_name = Path(str(selected["path"])).stem.removeprefix("vol_")
    return {
        "metrics": selected["metrics"],
        "path": str(auto_path),
        "selected_model": selected_name,
        "candidates": {
            Path(str(r["path"])).stem.removeprefix("vol_"): r["metrics"] for r in results
        },
    }


def _requested_garch_security_ids(
    security_ids: list[str] | tuple[str, ...] | None,
) -> list[str] | None:
    if security_ids is None:
        return None
    requested: list[str] = []
    seen: set[str] = set()
    for raw in security_ids:
        if isinstance(raw, bool) or not isinstance(raw, str):
            raise ValueError("per-security GARCH security_ids must be strings")
        sid = raw.strip()
        if not sid:
            raise ValueError("per-security GARCH security_ids must be non-empty")
        if sid in seen:
            raise ValueError("per-security GARCH security_ids must be unique")
        seen.add(sid)
        requested.append(sid)
    if not requested:
        raise ValueError("per-security GARCH security_ids must be non-empty")
    return requested


def garch_name_walk_forward(
    config: AppConfig,
    *,
    frame: Any | None = None,
    security_ids: list[str] | tuple[str, ...] | None = None,
    make_model: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Walk-forward QLIKE/density for the per-security GARCH namespace.

    Clones the training GARCH specification and refits each name on that name's
    strictly prior ``ret_1``. Metrics stamp ``scoring_scope=security_level_ret_1``
    and do not replace date-level ``train_volatility`` overlay scores,
    ``vol_20``, or ``max_predicted_vol``. Density targets are each name's
    origin ``ret_1``, never ``future_realized_var_h`` and never the
    equal-weight cross-section. Research-diagnostic only.
    """
    set_global_seed(config.train.random_seed)
    label = config.train.volatility_target
    df = frame if frame is not None else panel(config, label=label)
    requested = _requested_garch_security_ids(security_ids)
    if "security_id" not in df.columns:
        raise PointInTimeError("per-security GARCH frame missing security_id")
    _require_garch_security_keys(df)
    if requested is not None:
        present = {str(sid).strip() for sid in df["security_id"].to_list() if sid is not None}
        missing = [sid for sid in requested if sid not in present]
        if missing:
            raise ValueError(f"per-security GARCH has no rows for {missing[0]!r}")
        df = df.filter(pl.col("security_id").cast(pl.String).is_in(requested))
    x, y, dates, feats, ids = design_matrix(
        df, label, ["vol_20", "vol_ewma", "vol_parkinson", "ret_1", "vol_of_vol"]
    )
    if x.shape[0] == 0:
        raise ValueError("per-security GARCH walk-forward has no labeled rows")
    label_end_times = _aligned_label_end_times(df, label, feats)
    label_horizon = _label_horizon(label)

    def _make() -> Any:
        if make_model is not None:
            return make_model()
        return GARCHVol(
            p=config.train.garch_p,
            q=config.train.garch_q,
            dist=config.train.garch_dist.value,
            vol=config.train.garch_vol.value,
            series_scope=GARCH_SECURITY_LEVEL_SCOPE,
        )

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    eval_dates: list[np.ndarray] = []
    eval_ids: list[np.ndarray] = []
    fold_status: list[str] = []
    density_by_key: dict[tuple[str, Any], dict[str, Any]] = {}
    for train_mask, test_mask in _walk_forward_splits(
        dates,
        config,
        horizon_bars=label_horizon,
        label_end_times=label_end_times,
    ):
        if (
            not np.asarray(train_mask, dtype=bool).any()
            or not np.asarray(test_mask, dtype=bool).any()
        ):
            continue
        fold_predictions, statuses, fold_density = _garch_name_oos_predictions(
            _make,
            x,
            y,
            dates,
            ids,
            test_mask,
            label_horizon=label_horizon,
            return_frame=df,
        )
        if fold_predictions.size == 0:
            continue
        predictions.append(fold_predictions)
        fold_status.extend(statuses)
        targets.append(y[np.asarray(test_mask, dtype=bool)])
        eval_dates.append(np.asarray(dates)[np.asarray(test_mask, dtype=bool)])
        eval_ids.append(np.asarray(ids)[np.asarray(test_mask, dtype=bool)])
        for origin, record in fold_density.items():
            if origin in density_by_key:
                raise ValueError("per-security GARCH one-step density origin repeated across folds")
            density_by_key[origin] = record
    if not predictions:
        raise ValueError("walk-forward training produced no trainable/evaluable fold")
    pred = np.clip(np.concatenate(predictions), config.train.qlike_floor, None)
    yy = np.concatenate(targets)
    scored_ids = np.concatenate(eval_ids)
    scored_dates = np.concatenate(eval_dates)
    session_index = {date: i for i, date in enumerate(sorted(set(np.asarray(dates).tolist())))}
    scored = name_level_qlike(
        scored_ids,
        scored_dates,
        pred,
        yy,
        horizon_bars=label_horizon,
        floor=config.train.qlike_floor,
        session_index=session_index,
    )
    metrics: dict[str, Any] = {
        "qlike": float(scored["qlike"]),
        "qlike_overlapping_dates": float(scored["qlike_overlapping"]),
        "n_origins_nonoverlapping": int(scored["n_origins_nonoverlapping"]),
        "n_origins_overlapping": int(scored["n_origins_overlapping"]),
        "n_names": int(scored["n_names"]),
        "origin_stride": int(scored["horizon_bars"]),
        "scoring_scope": GARCH_SECURITY_LEVEL_SCOPE,
    }
    if density_by_key:
        origin_keys = sorted(density_by_key)
        density_metrics = name_level_one_step_density_summary(
            [key[0] for key in origin_keys],
            np.asarray([key[1] for key in origin_keys], dtype=object),
            np.asarray([density_by_key[key]["log_score"] for key in origin_keys], dtype=float),
            np.asarray([density_by_key[key]["crps"] for key in origin_keys], dtype=float),
            np.asarray([density_by_key[key]["pit"] for key in origin_keys], dtype=float),
            horizon_bars=label_horizon,
            session_index=session_index,
        )
        density_metrics.pop("scoring_scope", None)
        density_metrics.pop("horizon_bars", None)
        crps_methods = {str(density_by_key[key]["crps_method"]) for key in origin_keys}
        density_metrics["crps_method_one_step"] = (
            crps_methods.pop() if len(crps_methods) == 1 else "mixed"
        )
        metrics.update(density_metrics)
    else:
        metrics.update(
            {
                "log_score_one_step": float("nan"),
                "ignorance_one_step": float("nan"),
                "crps_one_step": float("nan"),
                "pit_ks_one_step": float("nan"),
                "pit_ks_p_one_step": float("nan"),
                "n_density_origins": 0,
                "n_density_origins_qlike_stride": 0,
                "log_score_one_step_qlike_origins": float("nan"),
                "crps_one_step_qlike_origins": float("nan"),
                "density_horizon": 1,
                "density_target": "security_level_ret_1",
            }
        )
    if fold_status:
        metrics["fallback_rate"] = float(np.mean(np.asarray(fold_status) != "fitted"))
    return {
        "metrics": metrics,
        "diagnostics": {
            "series_scope": GARCH_SECURITY_LEVEL_SCOPE,
            "oos_scoring": "security_level_nonoverlapping_qlike+one_step_density",
            "n_origins_nonoverlapping": int(metrics["n_origins_nonoverlapping"]),
            "n_origins_overlapping": int(metrics["n_origins_overlapping"]),
            "origin_stride": label_horizon,
            "n_density_origins": int(metrics["n_density_origins"]),
            "n_names": int(metrics["n_names"]),
            "density_target": "security_level_ret_1",
            "density_horizon": 1,
        },
    }


__all__ = [
    "garch_name_walk_forward",
    "train_volatility",
    "train_volatility_auto",
]
