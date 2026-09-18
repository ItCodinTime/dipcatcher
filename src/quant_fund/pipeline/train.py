"""Train forecast families on gold panels. Holdout is not used for Optuna."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.config.models import AppConfig
from quant_fund.metrics.cross_section import date_ic_series
from quant_fund.metrics.risk import losses_from_returns
from quant_fund.metrics.scoring import icir, mean_pinball, qlike, quantile_crossing_rate
from quant_fund.models.alpha import HistoricalMeanAlpha
from quant_fund.models.distribution import (
    EmpiricalDistribution,
    GaussianDistribution,
    LinearQuantileDistribution,
    TreeQuantileDistribution,
)
from quant_fund.models.ranking import (
    CompositeRanker,
    ElasticNetRanker,
    LGBMLambdaRanker,
    LGBMRegRanker,
    RidgeRanker,
    XGBRegRanker,
    group_sizes,
)
from quant_fund.models.regime import GaussianHMMRegime, SingleStateRegime, VolThresholdRegime
from quant_fund.models.tail import DrawdownClassifier, GaussianTail, HistoricalTail
from quant_fund.models.volatility import EWMAVol, GARCHVol, HARVol, RollingVol, TreeVol
from quant_fund.pipeline.dataset import design_matrix, panel
from quant_fund.registry.mlflow_store import (
    configure_tracking,
    log_run,
)
from quant_fund.utils.seeds import set_global_seed
from quant_fund.validation.purging import purge_mask
from quant_fund.validation.walk_forward import Fold, walk_forward


def _chronological_split(
    dates: Any,
    *,
    train_fraction: float = 0.7,
    horizon_bars: int = 1,
    embargo_bars: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split whole decision dates, purging labels and an explicit embargo.

    This compatibility helper is intentionally date-level: no panel rows from
    a date are split across train and test.  ``horizon_bars`` removes dates
    whose forward labels could reach the test boundary; ``embargo_bars`` adds
    the configured post-training gap independently of the label horizon.
    """
    values = np.asarray(dates)
    unique = sorted(set(values.tolist()))
    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if int(horizon_bars) < 0:
        raise ValueError("horizon_bars must be non-negative")
    if int(embargo_bars) < 0:
        raise ValueError("embargo_bars must be non-negative")
    if len(unique) < 2:
        raise ValueError("chronological_split requires at least 2 unique dates")
    cut = int(np.floor(len(unique) * float(train_fraction)))
    cut = min(max(cut, 1), len(unique) - 1)
    train_end = max(0, cut - int(horizon_bars) - int(embargo_bars))
    train_dates = set(unique[:train_end])
    test_dates = set(unique[cut:])
    return np.isin(values, list(train_dates)), np.isin(values, list(test_dates))


def _walk_forward_splits(
    dates: Any,
    config: Any,
    *,
    horizon_bars: int,
    label_end_times: Any | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return disjoint, date-level train/test masks with purging and embargo."""
    values = np.asarray(dates)
    ends = None if label_end_times is None else np.asarray(label_end_times)
    if ends is not None and len(ends) != len(values):
        raise ValueError("label_end_times must align with dates")
    unique = sorted(set(values.tolist()))
    validation = getattr(config, "validation", config)
    embargo_method = getattr(config, "embargo_bars", None)
    configured = (
        embargo_method() if callable(embargo_method) else getattr(validation, "embargo_bars", None)
    )
    # Explicit embargo_bars=0 must stay 0 (``or`` would substitute the horizon).
    embargo = int(configured) if configured is not None else int(horizon_bars)
    folds = walk_forward(
        values.tolist(),
        validation,
        horizon_bars=int(horizon_bars),
        embargo_bars=embargo,
        label_end_times=None if ends is None else ends.tolist(),
    )
    if not folds:
        # Short samples still need a genuine untouched test block. Purge the
        # candidate train dates before the test boundary instead of splitting
        # rows. When observed label endpoints are available, use them as the
        # source of truth; session arithmetic is only the compatibility fallback.
        cut = max(1, len(unique) // 2)
        test_times = unique[cut:]
        if ends is not None and test_times:
            end_by_time: dict[Any, Any] = {}
            for date, end in zip(values.tolist(), ends.tolist(), strict=True):
                previous = end_by_time.get(date)
                if previous is None or end > previous:
                    end_by_time[date] = end
            train_candidates = unique[:cut]
            session_idx = {date: i for i, date in enumerate(unique)}
            keep = purge_mask(
                train_candidates,
                test_times[0],
                test_times[-1],
                int(horizon_bars),
                session_index=session_idx,
                label_end_times=[end_by_time[date] for date in train_candidates],
            )
            train_times = [
                date
                for date, safe in zip(train_candidates, keep, strict=True)
                if safe and session_idx[date] + embargo < cut
            ]
        else:
            train_end = max(0, cut - int(horizon_bars) - embargo)
            train_times = unique[:train_end]
        folds = [Fold(train_times=train_times, val_times=[], test_times=test_times)]
    return [
        (
            np.isin(values, np.asarray(fold.train_times, dtype=object)),
            np.isin(values, np.asarray(fold.test_times, dtype=object)),
        )
        for fold in folds
        if fold.train_times and fold.test_times
    ]


def _split_fold(dates, x, y, fold):
    tr = np.isin(dates, np.array(fold.train_times, dtype=object))
    va = np.isin(dates, np.array(fold.val_times, dtype=object))
    te = np.isin(dates, np.array(fold.test_times, dtype=object))
    return tr, va, te


def _label_horizon(label: str, default: int = 5) -> int:
    """Extract the trailing bar horizon encoded by a forward label name."""
    match = re.search(r"_(\d+)$", label)
    return int(match.group(1)) if match else int(default)


def _aligned_label_end_times(frame: Any, label: str, features: list[str]) -> np.ndarray | None:
    """Return row-aligned observed label endpoints when the label engine provides them."""
    horizon = _label_horizon(label)
    endpoint = f"label_end_time_{horizon}"
    if endpoint not in frame.columns:
        return None
    # Match design_matrix's selected columns and null filtering exactly.  The
    # endpoint is non-null whenever a forward label is usable.
    sub = frame.select(["event_time", "security_id", label, *features, endpoint]).drop_nulls()
    return sub[endpoint].to_numpy()


def _require_model(name: str, catalog: set[str], family: str) -> str:
    """Fail closed on unknown model names before any panel I/O."""
    if name not in catalog:
        raise ValueError(f"unknown {family} model {name!r}")
    return name


def _garch_return_history(frame: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return finite equal-weight ``ret_1`` observations aggregated by date.

    This deliberately reads the full feature panel instead of the label-filtered
    design matrix: the latest returns are known even when their forward variance
    labels are structurally unavailable.
    """
    columns = set(frame.columns)
    if not {"event_time", "ret_1"}.issubset(columns):
        raise ValueError("GARCH requires event_time and ret_1 in the full feature panel")
    history = frame.select(["event_time", "ret_1"]).drop_nulls()
    raw_dates = np.asarray(history["event_time"].to_numpy())
    raw_values = np.asarray(history["ret_1"].to_numpy(), dtype=float)
    finite = np.isfinite(raw_values)
    if not np.any(finite):
        raise ValueError("GARCH return history contains no finite ret_1 observations")
    raw_dates = raw_dates[finite]
    raw_values = raw_values[finite]
    ordered_dates = sorted(set(raw_dates.tolist()))
    dates = np.asarray(ordered_dates)
    values = np.asarray(
        [np.mean(raw_values[raw_dates == date]) for date in ordered_dates], dtype=float
    )
    return dates, values


def train_ranking(config: AppConfig, model_name: str = "ridge") -> dict[str, Any]:
    _require_model(
        model_name,
        {"composite", "ridge", "elasticnet", "xgboost", "lightgbm", "lambdarank", "xendcg"},
        "ranking",
    )
    set_global_seed(config.train.random_seed)
    label = config.train.ranking_target
    df = panel(config, label=label)
    x, y, dates, feats, ids = design_matrix(df, label)
    label_end_times = _aligned_label_end_times(df, label, feats)
    horizon = _label_horizon(label)
    evaluation_scores: list[np.ndarray] = []
    evaluation_targets: list[np.ndarray] = []
    evaluation_dates: list[np.ndarray] = []
    last_model: Any = None
    # Evaluate every disjoint test block. Validation blocks are never used as
    # performance estimates, and each model is trained only on purged dates.
    for tr, te in _walk_forward_splits(
        dates, config, horizon_bars=horizon, label_end_times=label_end_times
    ):
        if not tr.any() or not te.any():
            continue
        model = _make_ranker(model_name, config)
        _fit_ranker(model, model_name, x[tr], y[tr], dates[tr])
        evaluation_scores.append(np.asarray(model.predict(x[te]), dtype=float))
        evaluation_targets.append(y[te])
        evaluation_dates.append(dates[te])
        last_model = model
    if last_model is None or not evaluation_scores:
        raise ValueError("walk-forward training produced no trainable/evaluable fold")
    date_metrics = date_ic_series(
        np.concatenate(evaluation_scores),
        np.concatenate(evaluation_targets),
        np.concatenate(evaluation_dates),
        min_names=4,
    )
    ics = date_metrics.pearson.tolist()
    rics = date_metrics.spearman.tolist()
    metrics = {
        "mean_ic": float(date_metrics.mean_pearson) if ics else float(np.nanmean(ics)),
        "mean_rank_ic": float(date_metrics.mean_spearman) if rics else float(np.nanmean(rics)),
        "icir": float(date_metrics.icir_pearson) if ics else icir(np.array(ics)),
        "ic_tstat": float(date_metrics.t_pearson),
        "ic_pvalue": float(date_metrics.p_pearson),
        "n_ic_dates": int(date_metrics.n_dates),
    }
    configure_tracking()
    run_id = log_run(
        family="ranking",
        name=model_name,
        params={"features": feats, "label": label, "model": model_name},
        metrics=metrics,
        tags={"git": "local", "data": config.data.source},
    )
    out = Path(config.data.root) / "metadata" / f"ranker_{model_name}.joblib"
    last_model.save(out)
    return {
        "metrics": metrics,
        "run_id": run_id,
        "path": str(out),
        "features": feats,
        "n": int(x.shape[0]),
    }


def _make_ranker(name: str, config: AppConfig) -> Any:
    t = config.train
    catalog = {
        "composite": CompositeRanker(),
        "ridge": RidgeRanker(t.ridge_alpha),
        "elasticnet": ElasticNetRanker(t.ridge_alpha, t.elasticnet_l1),
        "xgboost": XGBRegRanker(t.xgb_n_estimators, t.xgb_max_depth, t.random_seed),
        "lightgbm": LGBMRegRanker(t.lgbm_n_estimators, t.lgbm_num_leaves, t.random_seed),
        "lambdarank": LGBMLambdaRanker(t.lgbm_n_estimators, t.random_seed, "lambdarank"),
        "xendcg": LGBMLambdaRanker(t.lgbm_n_estimators, t.random_seed, "rank_xendcg"),
    }
    if name not in catalog:
        raise ValueError(f"unknown ranking model {name!r}")
    return catalog[name]


def _fit_ranker(model: Any, name: str, x, y, dates) -> None:
    if name in {"lambdarank", "xendcg"}:
        order = np.argsort(dates, kind="mergesort")
        model.fit(x[order], y[order], group=group_sizes(dates[order]))
    else:
        model.fit(x, y)


def train_distribution(config: AppConfig, model_name: str = "gaussian") -> dict[str, Any]:
    _require_model(
        model_name,
        {"empirical", "gaussian", "linear_qr", "xgboost", "lightgbm"},
        "distribution",
    )
    set_global_seed(config.train.random_seed)
    label = config.train.distribution_target
    df = panel(config, label=label)
    x, y, dates, feats, _ = design_matrix(df, label)
    label_end_times = _aligned_label_end_times(df, label, feats)
    taus = config.quantiles.levels

    def make_model() -> Any:
        catalog = {
            "empirical": EmpiricalDistribution(taus),
            "gaussian": GaussianDistribution(taus),
            "linear_qr": LinearQuantileDistribution(taus),
            "xgboost": TreeQuantileDistribution(taus, "xgboost", config.train.random_seed),
            "lightgbm": TreeQuantileDistribution(taus, "lightgbm", config.train.random_seed),
        }
        if model_name not in catalog:
            raise ValueError(f"unknown distribution model {model_name!r}")
        return catalog[model_name]

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for train_mask, test_mask in _walk_forward_splits(
        dates,
        config,
        horizon_bars=_label_horizon(label),
        label_end_times=label_end_times,
    ):
        if not train_mask.any() or not test_mask.any():
            continue
        fold_model = make_model().fit(x[train_mask], y[train_mask])
        predictions.append(np.asarray(fold_model.predict(x[test_mask]), dtype=float))
        targets.append(y[test_mask])
    if not predictions:
        raise ValueError("walk-forward training produced no trainable/evaluable fold")
    q = np.concatenate(predictions)
    yy = np.concatenate(targets)
    mid = taus.index(min(taus, key=lambda t: abs(t - 0.5))) if taus else 0
    pin = mean_pinball(yy, q[:, mid], taus[mid])
    cross = quantile_crossing_rate(q, np.array(taus))
    metrics = {"mean_pinball": pin, "crossing_rate": cross}
    # Refit the persisted artifact on every currently available labeled row;
    # only the fold predictions above are used for reported performance.
    model = make_model().fit(x, y)
    configure_tracking()
    run_id = log_run(
        family="distribution",
        name=model_name,
        params={"model": model_name},
        metrics=metrics,
        tags={"data": config.data.source},
    )
    path = Path(config.data.root) / "metadata" / f"dist_{model_name}.joblib"
    model.save(path)
    return {"metrics": metrics, "run_id": run_id, "path": str(path)}


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
) -> tuple[np.ndarray, list[str]]:
    """Forecast each test row from the last information available at its origin.

    GARCH is a stateful time-series model, so one forecast from the fold boundary
    cannot be repeated across a multi-date test window.  Refit on the expanding
    history strictly before each test date; this is causal and lets the return
    likelihood absorb information that was available before that origin.
    ``return_dates``/``return_values`` come from the full feature panel, not the
    label-filtered design matrix, so structurally unavailable forward labels cannot
    remove otherwise usable historical returns.
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
        return np.empty(0, dtype=float), []

    by_date: dict[Any, float] = {}
    statuses: list[str] = []
    for test_date in sorted(set(date_values[test_indices].tolist())):
        origin_mask = date_values < test_date
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

    predictions = np.asarray(
        [by_date[date_values[index].item()] for index in test_indices], dtype=float
    )
    return predictions, statuses


def train_volatility(config: AppConfig, model_name: str = "ewma") -> dict[str, Any]:
    _require_model(
        model_name,
        {"rolling", "ewma", "garch", "har", "xgboost", "lightgbm"},
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

    def make_model() -> Any:
        catalog = {
            "rolling": RollingVol(),
            "ewma": EWMAVol(config.features.ewma_lambda),
            "garch": GARCHVol(
                p=config.train.garch_p,
                q=config.train.garch_q,
                dist=config.train.garch_dist.value,
                vol=config.train.garch_vol.value,
                series_scope="date_level_equal_weight_cross_section",
            ),
            "har": HARVol(config.train.har_log),
            "xgboost": TreeVol("xgboost", config.train.random_seed),
            "lightgbm": TreeVol("lightgbm", config.train.random_seed),
        }
        if model_name not in catalog:
            raise ValueError(f"unknown volatility model {model_name!r}")
        return catalog[model_name]

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    fold_status: list[str] = []
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
            fold_predictions, statuses = _garch_oos_predictions(
                make_model,
                x,
                y,
                dates,
                test_mask,
                label_horizon=label_horizon,
                return_dates=return_dates,
                return_values=return_values,
            )
            predictions.append(fold_predictions)
            fold_status.extend(statuses)
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
    if not predictions:
        raise ValueError("walk-forward training produced no trainable/evaluable fold")
    pred = np.clip(np.concatenate(predictions), config.train.qlike_floor, None)
    yy = np.concatenate(targets)
    metrics = {"qlike": qlike(yy, pred, config.train.qlike_floor)}
    if model_name == "garch" and fold_status:
        metrics["fallback_rate"] = float(np.mean(np.asarray(fold_status) != "fitted"))
    if model_name == "garch":
        model = make_model().fit(x, y, returns=return_values)
    else:
        model = make_model().fit(x, y)
    configure_tracking()
    run_id = log_run(
        family="volatility",
        name=model_name,
        params={
            **(
                {
                    "garch_p": config.train.garch_p,
                    "garch_q": config.train.garch_q,
                    "garch_dist": config.train.garch_dist.value,
                    "garch_vol": config.train.garch_vol.value,
                    "target": label,
                    "target_contract": "forward_realized_variance_evaluation_only",
                    "fit_input": "ret_1_decimal_returns",
                    "forecast_horizon": label_horizon,
                    "variance_units": "decimal_squared",
                    "series_scope": "date_level_equal_weight_cross_section",
                }
                if model_name == "garch"
                else {}
            )
        },
        metrics=metrics,
        tags={"data": config.data.source},
    )
    path = Path(config.data.root) / "metadata" / f"vol_{model_name}.joblib"
    model.save(path)
    result = {"metrics": metrics, "run_id": run_id, "path": str(path)}
    if model_name == "garch":
        result["diagnostics"] = {
            "fit_status": model.fit_status,
            "converged": model.converged,
            "fallback_reason": model.fallback_reason,
            "n_obs": model.n_obs,
            "returns_scale": model.returns_scale,
            "series_scope": "date_level_equal_weight_cross_section",
        }
    return result


def train_alpha(config: AppConfig, model_name: str = "ridge") -> dict[str, Any]:
    _require_model(
        model_name,
        {"mean", "ridge", "elasticnet", "xgboost", "lightgbm", "lambdarank", "xendcg", "composite"},
        "alpha",
    )
    if model_name == "mean":
        # reuse ranking path with historical mean
        label = config.train.ranking_target
        df = panel(config, label=label)
        x, y, _, _, _ = design_matrix(df, label)
        m = HistoricalMeanAlpha().fit(x, y)
        path = Path(config.data.root) / "metadata" / "alpha_mean.joblib"
        m.save(path)
        return {"metrics": {"mean": float(m.mean_)}, "path": str(path)}
    return train_ranking(config, "ridge" if model_name == "ridge" else model_name)


def train_regime(config: AppConfig, model_name: str = "hmm") -> dict[str, Any]:
    _require_model(model_name, {"hmm", "threshold", "single", "single_state"}, "regime")
    df = panel(config)
    cols = [c for c in ["mkt_ret_1", "mkt_vol_20", "cs_dispersion", "breadth"] if c in df.columns]
    sub = df.select(["event_time", *cols]).unique("event_time").drop_nulls().sort("event_time")
    dates = sub["event_time"].to_numpy()
    x = sub.select(cols).to_numpy().astype(float)

    def make_model() -> Any:
        if model_name == "hmm":
            return GaussianHMMRegime(config.train.n_hmm_states, config.train.random_seed)
        if model_name == "threshold":
            return VolThresholdRegime()
        if model_name in {"single", "single_state"}:
            return SingleStateRegime()
        raise ValueError(f"unknown regime model {model_name!r}")

    heldout_ll: list[float] = []
    for train_mask, test_mask in _walk_forward_splits(dates, config, horizon_bars=1):
        if not train_mask.any() or not test_mask.any():
            continue
        fold_model = make_model().fit(x[train_mask])
        if model_name == "hmm":
            heldout_ll.append(float(fold_model.aic_bic(x[test_mask])["avg_ll"]))
    if not heldout_ll and model_name == "hmm":
        raise ValueError("walk-forward training produced no trainable/evaluable fold")

    # Persist a production model fit on all labeled history; all reported HMM
    # likelihood is from untouched fold test blocks, never this refit.
    m = make_model().fit(x)
    extra = {"oos_avg_ll": float(np.mean(heldout_ll))} if heldout_ll else {}
    path = Path(config.data.root) / "metadata" / f"regime_{model_name}.joblib"
    m.save(path)
    return {"metrics": extra, "path": str(path), "labels": getattr(m, "labels", {})}


def train_tail(config: AppConfig, model_name: str = "historical") -> dict[str, Any]:
    _require_model(model_name, {"historical", "gaussian", "drawdown"}, "tail")
    df = panel(config)
    return_labels = [c for c in df.columns if c.startswith("future_return")]
    if not return_labels:
        raise ValueError("gold panel has no future_return* label column for tail training")
    label = return_labels[0]
    if model_name == "drawdown":
        event_labels = [c for c in df.columns if c.startswith("future_tail_event")]
        label = event_labels[0] if event_labels else label
    x, y, dates, tail_features, _ = design_matrix(df, label)
    label_end_times = _aligned_label_end_times(df, label, tail_features)

    def make_model() -> Any:
        if model_name == "gaussian":
            return GaussianTail()
        if model_name == "drawdown":
            return DrawdownClassifier()
        if model_name == "historical":
            return HistoricalTail()
        raise ValueError(f"unknown tail model {model_name!r}")

    breach_rates: list[float] = []
    es_excessions: list[float] = []
    brier_scores: list[float] = []
    for train_mask, test_mask in _walk_forward_splits(
        dates,
        config,
        horizon_bars=_label_horizon(label),
        label_end_times=label_end_times,
    ):
        if not train_mask.any() or not test_mask.any():
            continue
        fold_model = make_model().fit(x[train_mask], y[train_mask])
        if model_name == "drawdown":
            pred = np.asarray(fold_model.predict(x[test_mask]), dtype=float)
            brier_scores.append(float(np.mean((pred - (y[test_mask] > 0.5)) ** 2)))
            continue
        var, es = fold_model.predict_var_es()
        losses = losses_from_returns(y[test_mask])
        breaches = losses >= var
        breach_rates.append(float(np.mean(breaches)))
        es_losses = losses[breaches]
        if es_losses.size:
            es_excessions.append(float(np.mean(es_losses) - es))
    if not (breach_rates or brier_scores):
        raise ValueError("walk-forward training produced no trainable/evaluable fold")

    # Persist a final estimator fit on all available history; metrics remain OOS.
    m = make_model().fit(x, y)
    path = Path(config.data.root) / "metadata" / f"tail_{model_name}.joblib"
    m.save(path)
    metrics: dict[str, float] = {}
    if breach_rates:
        var, es = m.predict_var_es()
        metrics = {
            "oos_var_breach_rate": float(np.mean(breach_rates)),
            "oos_es_excess": float(np.mean(es_excessions)) if es_excessions else float("nan"),
            "var": var,
            "es": es,
        }
    else:
        metrics = {"oos_brier": float(np.mean(brier_scores))}
    return {"metrics": metrics, "path": str(path)}


def train_family(config: AppConfig, family: str, model_name: str | None = None) -> dict[str, Any]:
    dispatch = {
        "ranking": lambda: train_ranking(config, model_name or "ridge"),
        "alpha": lambda: train_alpha(config, model_name or "ridge"),
        "distribution": lambda: train_distribution(config, model_name or "gaussian"),
        "volatility": lambda: train_volatility(config, model_name or "ewma"),
        "regime": lambda: train_regime(config, model_name or "hmm"),
        "tail": lambda: train_tail(config, model_name or "historical"),
        "covariance": lambda: {"metrics": {}, "note": "covariance is estimated at forecast time"},
        "liquidity": lambda: {"metrics": {}, "note": "liquidity uses parameterized cost model"},
    }
    if family not in dispatch:
        raise ValueError(f"unknown family {family}")
    result = dispatch[family]()
    if config.data.source == "synthetic":
        result["data_source"] = "SYNTHETIC"
    return result
