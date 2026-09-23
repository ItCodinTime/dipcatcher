"""Train forecast families on gold panels. Holdout is not used for Optuna."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.lightspeed.ranker import NauticaRanker
from quant_fund.metrics.cross_section import date_ic_series
from quant_fund.metrics.probability import brier_score
from quant_fund.metrics.risk import losses_from_returns
from quant_fund.metrics.scoring import (
    GARCH_ONE_STEP_CRPS_TAUS,
    crps_from_quantiles,
    crps_gaussian,
    icir,
    mean_pinball,
    name_level_one_step_density_summary,
    name_level_qlike,
    one_step_density_summary,
    overlap_aware_qlike,
    qlike,
    quantile_crossing_rate,
)
from quant_fund.models.alpha import HistoricalMeanAlpha
from quant_fund.models.asset_pricing import IPCARanker, RandomFourierRanker, SDFRidgeRanker
from quant_fund.models.base import (
    JoblibMixin,
    artifact_identity,
    load_joblib_artifact,
    save_joblib_artifact,
)
from quant_fund.models.calibration import ProbabilityCalibrator
from quant_fund.models.cs_papers import (
    DATED_FIT_RANKERS,
    DATED_PREDICT_RANKERS,
    ID_FIT_RANKERS,
    ID_PREDICT_RANKERS,
    PAPER_RANKER_NAMES,
    AdaptiveLassoRanker,
    ClassicRanker,
    ClassicShortRanker,
    CombinationRanker,
    DoubleSelectionRanker,
    FamaMacBethRanker,
    FamaMacBethRidgeRanker,
    FNWRanker,
    GBRTRanker,
    GXThreePassRanker,
    ICWeightedCombinationRanker,
    KraussRanker,
    MSFECombinationRanker,
    PCRRanker,
    PLSRanker,
    PrincipalPortfolioRanker,
    ReversalRanker,
    RPPCARanker,
    SDFElasticNetRanker,
    ThreePassFilterRanker,
    TSMOMRanker,
    VMERanker,
    make_combo_ic_st,
    make_fm_st,
    make_ridge_neut,
    make_ridge_st,
)
from quant_fund.models.deep_rl import PolicyGradientRanker, run_policy_gradient_panel
from quant_fund.models.distribution import (
    EmpiricalDistribution,
    GaussianDistribution,
    LinearQuantileDistribution,
    TreeQuantileDistribution,
)
from quant_fund.models.quantile_bandit import QuantileThompson
from quant_fund.models.ranking import (
    CompositeRanker,
    ElasticNetRanker,
    EnsembleRanker,
    LGBMLambdaRanker,
    LGBMRegRanker,
    NeuralRanker,
    RidgeRanker,
    XGBRegRanker,
    group_sizes,
)
from quant_fund.models.realized_garch import (
    REALIZED_GARCH_MEASURE,
    RealizedGARCHVol,
    parkinson_daily_variance,
)
from quant_fund.models.regime import GaussianHMMRegime, SingleStateRegime, VolThresholdRegime
from quant_fund.models.rl import (
    LinearThompsonRanker,
    LinUCBRanker,
    run_linucb_panel,
    run_thompson_panel,
)
from quant_fund.models.tail import DrawdownClassifier, GaussianTail, HistoricalTail
from quant_fund.models.volatility import (
    GARCH_DATE_LEVEL_SCOPE,
    GARCH_SECURITY_LEVEL_SCOPE,
    EWMAVol,
    GARCHVol,
    HARVol,
    RollingVol,
    TreeVol,
)
from quant_fund.pipeline.dataset import design_frame, design_matrix, panel
from quant_fund.registry.mlflow_store import (
    attach_artifact_identity,
    configure_tracking,
    log_run,
)
from quant_fund.reporting.report import write_evidence_report
from quant_fund.schemas.errors import PointInTimeError
from quant_fund.utils.hashing import canonical_frame_fingerprint, canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision, git_worktree_sha256
from quant_fund.utils.seeds import set_global_seed
from quant_fund.validation.purging import purge_mask
from quant_fund.validation.walk_forward import Fold, timestamp_ns, walk_forward

RANKING_MODEL_NAMES = {
    "composite",
    "ridge",
    "elasticnet",
    "neural",
    "ensemble",
    "xgboost",
    "lightgbm",
    "lambdarank",
    "xendcg",
    *PAPER_RANKER_NAMES,
}


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
    date_ns = timestamp_ns(values)
    return (
        np.isin(date_ns, timestamp_ns(list(train_dates))),
        np.isin(date_ns, timestamp_ns(list(test_dates))),
    )


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
    date_ns = timestamp_ns(values)
    return [
        (
            np.isin(date_ns, timestamp_ns(fold.train_times)),
            np.isin(date_ns, timestamp_ns(fold.test_times)),
        )
        for fold in folds
        if fold.train_times and fold.test_times
    ]


def _split_fold(dates, x, y, fold):
    date_ns = timestamp_ns(dates)
    tr = np.isin(date_ns, timestamp_ns(fold.train_times))
    va = np.isin(date_ns, timestamp_ns(fold.val_times))
    te = np.isin(date_ns, timestamp_ns(fold.test_times))
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
    sub = design_frame(frame, label, features, extra_columns=[endpoint])
    return sub[endpoint].to_numpy()


def _require_model(name: str, catalog: set[str], family: str) -> str:
    """Fail closed on unknown model names before any panel I/O."""
    if name not in catalog:
        raise ValueError(f"unknown {family} model {name!r}")
    return name


def _stamp_strictly_before(stamp: object, asof: object) -> bool:
    """Compare panel timestamps to asof without mixing naive/aware datetimes."""
    if isinstance(stamp, datetime) and isinstance(asof, datetime):
        if stamp.tzinfo is None and asof.tzinfo is not None:
            return stamp.replace(tzinfo=asof.tzinfo) < asof
        if stamp.tzinfo is not None and asof.tzinfo is None:
            return stamp.replace(tzinfo=None) < asof
        return stamp < asof
    return bool(stamp < asof)  # type: ignore[operator]


def _stamp_at_or_before(stamp: object, asof: object) -> bool:
    """True when ``stamp`` is observable at the decision origin."""
    if isinstance(stamp, datetime) and isinstance(asof, datetime):
        if stamp.tzinfo is None and asof.tzinfo is not None:
            return stamp.replace(tzinfo=asof.tzinfo) <= asof
        if stamp.tzinfo is not None and asof.tzinfo is None:
            return stamp.replace(tzinfo=None) <= asof
        return stamp <= asof
    return bool(stamp <= asof)  # type: ignore[operator]


def _available_stamp_is_missing(stamp: object) -> bool:
    if stamp is None:
        return True
    if isinstance(stamp, datetime):
        return False
    try:
        if stamp != stamp:  # NaT / NaN
            return True
    except (TypeError, ValueError):
        return True
    return False


def _require_garch_security_keys(frame: Any) -> None:
    """Fail closed on blank ids or duplicate ``(security_id, event_time)`` keys.

    Equal-weight date-level means and per-name histories are undefined when a
    name is double-counted. Frames without ``security_id`` keep the legacy
    date-only path used by univariate fixtures.
    """
    columns = set(frame.columns)
    if "security_id" not in columns or "event_time" not in columns:
        return
    if frame.is_empty():
        return
    blank = frame.filter(
        pl.col("security_id").is_null()
        | (pl.col("security_id").cast(pl.String).str.strip_chars() == "")
    )
    if blank.height:
        raise PointInTimeError("GARCH return history contains blank security_id")
    if frame.select(["security_id", "event_time"]).is_duplicated().any():
        raise PointInTimeError(
            "GARCH return history contains duplicate security_id/event_time rows"
        )


def _garch_name_return_history(
    frame: Any,
    security_id: str,
    asof: object | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal ``ret_1`` history for one security, never a pooled cross-section.

    Duplicate panel keys fail closed on the full frame so a colliding name
    cannot silently inflate another name's series. ``available_time`` uses the
    same as-of contract as the date-level overlay.
    """
    if not isinstance(security_id, str) or not security_id.strip():
        raise ValueError("per-security GARCH requires a non-empty security_id")
    columns = set(frame.columns)
    if "security_id" not in columns:
        raise PointInTimeError("per-security GARCH frame missing security_id")
    _require_garch_security_keys(frame)
    sid = security_id.strip()
    name_frame = frame.filter(pl.col("security_id").cast(pl.String) == sid)
    if name_frame.is_empty():
        return np.asarray([]), np.asarray([], dtype=float)
    return _garch_return_history(name_frame, asof=asof)


def _garch_return_history(
    frame: Any,
    asof: object | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite equal-weight ``ret_1`` observations aggregated by date.

    This deliberately reads the full feature panel instead of the label-filtered
    design matrix: the latest returns are known even when their forward variance
    labels are structurally unavailable. When ``asof`` is supplied, only rows
    with ``event_time < asof`` enter the series. A present ``available_time``
    column is PIT-filtered at the same origin (``available_time <= asof``);
    null availability among otherwise usable rows fails closed rather than
    treating an unpublished restatement as observable. Frames without
    ``available_time`` keep the legacy event-time path.
    """
    _require_garch_security_keys(frame)
    columns = set(frame.columns)
    if not {"event_time", "ret_1"}.issubset(columns):
        raise ValueError("GARCH requires event_time and ret_1 in the full feature panel")
    has_availability = "available_time" in columns
    selected = ["event_time", "ret_1"]
    if has_availability:
        selected.append("available_time")
    history = frame.select(selected).drop_nulls(subset=["event_time", "ret_1"])
    raw_dates = np.asarray(history["event_time"].to_numpy())
    raw_values = np.asarray(history["ret_1"].to_numpy(), dtype=float)
    finite = np.isfinite(raw_values)
    if not np.any(finite):
        raise ValueError("GARCH return history contains no finite ret_1 observations")
    raw_dates = raw_dates[finite]
    raw_values = raw_values[finite]
    raw_available = (
        np.asarray(history["available_time"].to_numpy())[finite] if has_availability else None
    )
    if asof is not None:
        keep = np.array(
            [_stamp_strictly_before(stamp, asof) for stamp in raw_dates.tolist()],
            dtype=bool,
        )
        if raw_available is not None:
            available_stamps = raw_available.tolist()
            missing = [
                _available_stamp_is_missing(stamp)
                for stamp, origin_ok in zip(available_stamps, keep.tolist(), strict=True)
                if origin_ok
            ]
            if any(missing):
                raise PointInTimeError(
                    "GARCH return history has null available_time; "
                    "refusing unobservable market overlay"
                )
            keep &= np.array(
                [
                    (not _available_stamp_is_missing(stamp)) and _stamp_at_or_before(stamp, asof)
                    for stamp in available_stamps
                ],
                dtype=bool,
            )
        raw_dates = raw_dates[keep]
        raw_values = raw_values[keep]
        if raw_dates.size == 0:
            return np.asarray([]), np.asarray([], dtype=float)
    ordered_dates = sorted(set(raw_dates.tolist()))
    dates = np.asarray(ordered_dates)
    values = np.asarray(
        [np.mean(raw_values[raw_dates == date]) for date in ordered_dates], dtype=float
    )
    return dates, values


def _realized_garch_ohlc_columns(frame: Any) -> tuple[str, str]:
    """Prefer split-adjusted high/low; never invent a close-to-close RV proxy."""
    columns = set(frame.columns)
    if {"high_split_adjusted", "low_split_adjusted"}.issubset(columns):
        return "high_split_adjusted", "low_split_adjusted"
    if {"high", "low"}.issubset(columns):
        return "high", "low"
    raise PointInTimeError(
        "Realized GARCH requires daily OHLC high/low; refusing close-to-close squared proxy"
    )


def _realized_garch_history(
    frame: Any,
    asof: object | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Date-level equal-weight ``ret_1`` paired with same-name Parkinson variance.

    One-day Parkinson is computed from daily OHLC. The 20-day ``vol_parkinson``
    feature is never the realized measure. Names missing a finite return or
    Parkinson drop from *both* means so the pair stays aligned. PIT filters
    match ``_garch_return_history``.
    """
    _require_garch_security_keys(frame)
    columns = set(frame.columns)
    if not {"event_time", "ret_1"}.issubset(columns):
        raise ValueError("Realized GARCH requires event_time and ret_1")
    high_col, low_col = _realized_garch_ohlc_columns(frame)
    has_availability = "available_time" in columns
    selected = ["event_time", "ret_1", high_col, low_col]
    if has_availability:
        selected.append("available_time")
    history = frame.select(selected).drop_nulls(subset=["event_time", "ret_1", high_col, low_col])
    if history.is_empty():
        raise ValueError("Realized GARCH history contains no finite OHLC/return pairs")
    park = parkinson_daily_variance(history[high_col].to_numpy(), history[low_col].to_numpy())
    raw_dates = np.asarray(history["event_time"].to_numpy())
    raw_values = np.asarray(history["ret_1"].to_numpy(), dtype=float)
    finite = np.isfinite(raw_values) & np.isfinite(park) & (park > 0.0)
    if not np.any(finite):
        raise ValueError("Realized GARCH history contains no finite Parkinson/return pairs")
    raw_dates = raw_dates[finite]
    raw_values = raw_values[finite]
    raw_park = park[finite]
    raw_available = (
        np.asarray(history["available_time"].to_numpy())[finite] if has_availability else None
    )
    if asof is not None:
        keep = np.array(
            [_stamp_strictly_before(stamp, asof) for stamp in raw_dates.tolist()],
            dtype=bool,
        )
        if raw_available is not None:
            available_stamps = raw_available.tolist()
            missing = [
                _available_stamp_is_missing(stamp)
                for stamp, origin_ok in zip(available_stamps, keep.tolist(), strict=True)
                if origin_ok
            ]
            if any(missing):
                raise PointInTimeError(
                    "Realized GARCH history has null available_time; "
                    "refusing unobservable Parkinson overlay"
                )
            keep &= np.array(
                [
                    (not _available_stamp_is_missing(stamp)) and _stamp_at_or_before(stamp, asof)
                    for stamp in available_stamps
                ],
                dtype=bool,
            )
        raw_dates = raw_dates[keep]
        raw_values = raw_values[keep]
        raw_park = raw_park[keep]
        if raw_dates.size == 0:
            return np.asarray([]), np.asarray([], dtype=float), np.asarray([], dtype=float)
    ordered_dates = sorted(set(raw_dates.tolist()))
    dates = np.asarray(ordered_dates)
    values = np.asarray(
        [np.mean(raw_values[raw_dates == date]) for date in ordered_dates], dtype=float
    )
    measures = np.asarray(
        [np.mean(raw_park[raw_dates == date]) for date in ordered_dates], dtype=float
    )
    return dates, values, measures


def train_ranking(
    config: AppConfig,
    model_name: str = "ridge",
    *,
    frame: pl.DataFrame | None = None,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    if model_name == "auto":
        return train_ranking_auto(config)
    _require_model(model_name, RANKING_MODEL_NAMES, "ranking")
    set_global_seed(config.train.random_seed)
    label = config.train.ranking_target
    df = frame if frame is not None else panel(config, label=label)
    x, y, dates, feats, ids = design_matrix(df, label, feature_names=feature_names)
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
        _fit_ranker(model, model_name, x[tr], y[tr], dates[tr], ids[tr], features=feats)
        evaluation_scores.append(_predict_ranker(model, model_name, x[te], dates[te], ids[te]))
        evaluation_targets.append(y[te])
        evaluation_dates.append(dates[te])
        last_model = model
    if last_model is None or not evaluation_scores:
        raise ValueError("walk-forward training produced no trainable/evaluable fold")
    # Persist the exact training order alongside the estimator. Forecasting
    # must not reconstruct this contract from whatever columns happen to be
    # present in a later panel.
    last_model.features = list(feats)
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


def train_ranking_auto(config: AppConfig) -> dict[str, Any]:
    """Select a supervised ranker using the existing purged walk-forward metric.

    Auto-selection is deliberately limited to deterministic, portable models;
    each candidate is evaluated by ``train_ranking`` and therefore retains the
    same causal folds and artifact checksums. The selected artifact is the
    returned deployment candidate, while the other candidate artifacts remain
    available for audit comparison.
    """
    candidates = ("ridge", "elasticnet", "neural", "ensemble")
    results = [train_ranking(config, candidate) for candidate in candidates]
    viable = [
        result
        for result in results
        if np.isfinite(float(result["metrics"].get("mean_ic", np.nan)))
    ]
    if not viable:
        raise ValueError("automatic ranking selection produced no finite candidate metric")
    selected = max(
        viable,
        key=lambda result: (
            float(result["metrics"].get("mean_ic", -np.inf)),
            float(result["metrics"].get("mean_rank_ic", -np.inf)),
        ),
    )
    selected_model = JoblibMixin.load(Path(str(selected["path"])))
    auto_path = Path(config.data.root) / "metadata" / "ranker_auto.joblib"
    selected_model.save(auto_path)
    return {
        **selected,
        "path": str(auto_path),
        "model": "auto",
        "selected_model": Path(str(selected["path"])).stem.removeprefix("ranker_"),
        "selection_metric": "mean_ic",
        "candidates": {
            Path(str(result["path"])).stem.removeprefix("ranker_"): result["metrics"]
            for result in results
        },
    }


def train_calibration(config: AppConfig, model_name: str = "isotonic") -> dict[str, Any]:
    """Fit a causal probability calibrator for the bounded momentum score."""
    if model_name == "auto":
        return train_calibration_auto(config)
    if model_name not in {"isotonic", "platt"}:
        raise ValueError(f"unknown calibration model {model_name!r}")
    label = config.train.ranking_target
    df = panel(config, label=label)
    required = {"event_time", "cs_pct_mom_20", label}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"calibration requires columns: {missing}")
    raw = df.select(["event_time", "cs_pct_mom_20", label]).drop_nulls()
    dates = np.asarray(raw["event_time"].to_numpy())
    scores = np.asarray(raw["cs_pct_mom_20"].to_numpy(), dtype=float)
    labels = (np.asarray(raw[label].to_numpy(), dtype=float) > 0.0).astype(float)
    finite = np.isfinite(scores) & np.isfinite(labels)
    scores, labels, dates = scores[finite], labels[finite], dates[finite]
    if scores.size < 10 or np.unique(labels).size < 2:
        raise ValueError("calibration requires >=10 finite rows and both label classes")
    oos_prob: list[np.ndarray] = []
    oos_label: list[np.ndarray] = []
    for train_mask, test_mask in _walk_forward_splits(
        dates, config, horizon_bars=_label_horizon(label)
    ):
        calibrator = ProbabilityCalibrator(model_name).fit(scores[train_mask], labels[train_mask])
        if not calibrator.fitted:
            continue
        oos_prob.append(calibrator.predict(scores[test_mask]))
        oos_label.append(labels[test_mask])
    final = ProbabilityCalibrator(model_name).fit(scores, labels)
    if not final.fitted:
        raise ValueError("calibration produced an unfitted artifact")
    metrics = {
        "oos_brier": float(brier_score(np.concatenate(oos_prob), np.concatenate(oos_label)))
        if oos_prob
        else float("nan"),
        "n_rows": float(scores.size),
        "n_oos_rows": float(sum(values.size for values in oos_label)),
    }
    configure_tracking()
    run_id = log_run(
        family="calibration",
        name=model_name,
        params={"score": "cs_pct_mom_20", "label": label, "model": model_name},
        metrics=metrics,
        tags={"data": config.data.source, "claim": "research_only"},
    )
    path = Path(config.data.root) / "metadata" / f"calibrator_{model_name}.joblib"
    final.score_feature = "cs_pct_mom_20"
    final.label = label
    final.horizon = label
    final.fit_start = str(dates.min())
    final.fit_end = str(dates.max())
    if oos_label:
        oos_dates = np.concatenate(
            [dates[test_mask] for _train_mask, test_mask in _walk_forward_splits(
                dates, config, horizon_bars=_label_horizon(label)
            ) if test_mask.any()]
        )
        if oos_dates.size:
            final.oos_start = str(oos_dates.min())
            final.oos_end = str(oos_dates.max())
    final.save(path)
    return {"metrics": metrics, "run_id": run_id, "path": str(path), "research_only": True}


def train_calibration_auto(config: AppConfig) -> dict[str, Any]:
    """Select isotonic or Platt calibration by lowest finite OOS Brier score."""
    results = [train_calibration(config, method) for method in ("isotonic", "platt")]
    viable = [
        result
        for result in results
        if np.isfinite(float(result["metrics"].get("oos_brier", np.nan)))
    ]
    if not viable:
        raise ValueError("automatic calibration selection produced no finite Brier score")
    selected = min(viable, key=lambda result: float(result["metrics"]["oos_brier"]))
    selected_model = JoblibMixin.load(Path(str(selected["path"])))
    auto_path = Path(config.data.root) / "metadata" / "calibrator_auto.joblib"
    selected_model.save(auto_path)
    selected_name = Path(str(selected["path"])).stem.removeprefix("calibrator_")
    return {
        **selected,
        "path": str(auto_path),
        "model": "auto",
        "selected_model": selected_name,
        "selection_metric": "oos_brier",
        "candidates": {
            Path(str(result["path"])).stem.removeprefix("calibrator_"): result["metrics"]
            for result in results
        },
    }


def _make_ranker(name: str, config: AppConfig) -> Any:
    t = config.train
    catalog = {
        "composite": CompositeRanker(),
        "ridge": RidgeRanker(t.ridge_alpha),
        "elasticnet": ElasticNetRanker(t.ridge_alpha, t.elasticnet_l1),
        "neural": NeuralRanker(t.random_seed),
        "ensemble": EnsembleRanker(t.random_seed),
        "xgboost": XGBRegRanker(t.xgb_n_estimators, t.xgb_max_depth, t.random_seed),
        "lightgbm": LGBMRegRanker(t.lgbm_n_estimators, t.lgbm_num_leaves, t.random_seed),
        "lambdarank": LGBMLambdaRanker(t.lgbm_n_estimators, t.random_seed, "lambdarank"),
        "xendcg": LGBMLambdaRanker(t.lgbm_n_estimators, t.random_seed, "rank_xendcg"),
        "rff": RandomFourierRanker(t.rff_n_features, t.rff_gamma, t.rff_z, t.random_seed),
        "rff_ridgeless": RandomFourierRanker(t.rff_n_features, t.rff_gamma, 0.0, t.random_seed),
        "sdf_ridge": SDFRidgeRanker(t.sdf_ridge_z),
        "sdf_en": SDFElasticNetRanker(t.sdf_en_l2, t.sdf_en_l1),
        "ipca": IPCARanker(t.ipca_n_factors, t.ipca_max_iter, t.ipca_tol, unrestricted=False),
        "ipca_alpha": IPCARanker(t.ipca_n_factors, t.ipca_max_iter, t.ipca_tol, unrestricted=True),
        "rp_pca": RPPCARanker(t.rp_pca_n_factors, t.rp_pca_gamma),
        "fnw": FNWRanker(t.fnw_n_intervals, t.fnw_lam),
        "gx3pass": GXThreePassRanker(t.gx_n_factors),
        "ds_lasso": DoubleSelectionRanker(t.ds_lasso_alpha),
        "fm": FamaMacBethRanker(),
        "pcr": PCRRanker(t.pcr_n_factors),
        "pls": PLSRanker(t.pls_n_factors),
        "tprf": ThreePassFilterRanker(t.tprf_n_factors),
        "gbrt": GBRTRanker(
            t.gbrt_n_estimators, t.gbrt_max_depth, t.gbrt_learning_rate, t.random_seed
        ),
        "pp": PrincipalPortfolioRanker(t.pp_n_factors),
        "combo": CombinationRanker(),
        "alasso": AdaptiveLassoRanker(t.alasso_alpha),
        "classic": ClassicRanker(),
        "fm_ridge": FamaMacBethRidgeRanker(t.fm_ridge_alpha),
        "combo_ic": ICWeightedCombinationRanker(),
        "reversal": ReversalRanker(),
        "classic_st": ClassicShortRanker(),
        "ridge_st": make_ridge_st(t.ridge_alpha),
        "ridge_neut": make_ridge_neut(t.ridge_alpha),
        "fm_st": make_fm_st(t.fm_ridge_alpha),
        "combo_ic_st": make_combo_ic_st(),
        "combo_msfe": MSFECombinationRanker(),
        "nautica": NauticaRanker(),
        "tsmom": TSMOMRanker(),
        "vme": VMERanker(),
        "krauss": KraussRanker(),
    }
    if name not in catalog:
        raise ValueError(f"unknown ranking model {name!r}")
    return catalog[name]


def _fit_ranker(model: Any, name: str, x, y, dates, ids=None, features=None) -> None:
    extra = {} if features is None else {"features": features}
    if name in {"lambdarank", "xendcg"}:
        order = np.argsort(dates, kind="mergesort")
        model.fit(x[order], y[order], group=group_sizes(dates[order]), **extra)
    elif name in ID_FIT_RANKERS:
        model.fit(x, y, dates=dates, ids=ids, **extra)
    elif name in DATED_FIT_RANKERS:
        model.fit(x, y, dates=dates, **extra)
    else:
        model.fit(x, y, **extra)


def _predict_ranker(model: Any, name: str, x, dates, ids=None) -> np.ndarray:
    if name in ID_PREDICT_RANKERS:
        return np.asarray(model.predict(x, dates=dates, ids=ids), dtype=float)
    if name in DATED_PREDICT_RANKERS:
        return np.asarray(model.predict(x, dates=dates), dtype=float)
    return np.asarray(model.predict(x), dtype=float)


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
    metrics = {"mean_pinball": pin, "crossing_rate": cross, "n_oos_rows": float(yy.size)}
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


def train_distribution_auto(config: AppConfig) -> dict[str, Any]:
    """Select a distribution by finite causal walk-forward pinball loss."""
    candidates = ["empirical", "gaussian", "linear_qr", "xgboost", "lightgbm"]
    results = [train_distribution(config, name) for name in candidates]
    eligible = [
        r
        for r in results
        if np.isfinite(float(r["metrics"].get("mean_pinball", np.nan)))
        and float(r["metrics"].get("n_oos_rows", 0.0)) >= config.train.auto_min_oos_rows
    ]
    if not eligible:
        raise ValueError("automatic distribution selection produced no finite candidate metric")
    selected = min(eligible, key=lambda r: float(r["metrics"]["mean_pinball"]))
    payload = load_joblib_artifact(Path(str(selected["path"])))
    auto_path = Path(config.data.root) / "metadata" / "dist_auto.joblib"
    save_joblib_artifact(payload, auto_path)
    selected_name = Path(str(selected["path"])).stem.removeprefix("dist_")
    return {
        "metrics": selected["metrics"],
        "path": str(auto_path),
        "selected_model": selected_name,
        "candidates": {Path(str(r["path"])).stem.removeprefix("dist_"): r["metrics"] for r in results},
    }


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
    dummy_x = np.zeros((0, 1), dtype=float)
    dummy_y = np.zeros(0, dtype=float)
    for index in test_indices.tolist():
        sid = _garch_name_row_id(id_values.tolist()[int(index)])
        test_date = date_values.tolist()[int(index)]
        key = (sid, test_date)
        if key in by_key:
            raise ValueError("per-security GARCH walk-forward has duplicate security_id/event_time")
        _hist_dates, historical_returns = _garch_name_return_history(
            return_frame, sid, asof=test_date
        )
        if historical_returns.size == 0:
            raise ValueError(
                f"per-security GARCH test origin has no strictly prior returns for {sid!r}"
            )
        model = make_model()
        model.fit(dummy_x, dummy_y, returns=historical_returns)
        forecast = model.forecast(horizon=label_horizon)
        cumulative = np.asarray(forecast["cumulative_variance"], dtype=float).reshape(-1)
        if cumulative.size < label_horizon or not np.isfinite(cumulative[label_horizon - 1]):
            raise ValueError("GARCH produced an invalid out-of-sample forecast")
        by_key[key] = float(cumulative[label_horizon - 1])
        statuses.append(str(getattr(model, "fit_status", "unknown")))
        if hasattr(model, "log_density") and hasattr(model, "pit"):
            if key not in origin_returns:
                raise ValueError("per-security GARCH one-step density target missing origin ret_1")
            density_by_key[key] = _garch_origin_density_record(model, origin_returns[key], forecast)

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
        catalog = {
            "rolling": RollingVol(),
            "ewma": EWMAVol(config.features.ewma_lambda),
            "garch": GARCHVol(
                p=config.train.garch_p,
                q=config.train.garch_q,
                dist=config.train.garch_dist.value,
                vol=config.train.garch_vol.value,
                series_scope=GARCH_DATE_LEVEL_SCOPE,
            ),
            "realized_garch": RealizedGARCHVol(
                series_scope=GARCH_DATE_LEVEL_SCOPE,
                realized_measure=REALIZED_GARCH_MEASURE,
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
        "candidates": {Path(str(r["path"])).stem.removeprefix("vol_"): r["metrics"] for r in results},
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


def train_alpha(config: AppConfig, model_name: str = "ridge") -> dict[str, Any]:
    _require_model(
        model_name,
        {"mean", *RANKING_MODEL_NAMES},
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


def train_reinforcement(config: AppConfig, model_name: str = "linucb") -> dict[str, Any]:
    """Train the paper-only contextual ranking policy on a causal gold panel.

    This is an online contextual-bandit evaluation, not a backtest: rewards are
    forecast targets and the trace never represents executable P&L.  The policy
    sees each decision date once, selects top-k rows, then updates only from the
    realized labels for that date.  The persisted artifact is the final policy
    state and all reported metrics are descriptive research diagnostics.
    """
    if model_name == "auto":
        return train_reinforcement_auto(config)
    if model_name == "policy_gradient":
        return train_policy_gradient(config)
    if model_name == "quantile_thompson":
        return train_quantile_bandit(config)
    if model_name not in {"linucb", "thompson"}:
        raise ValueError(f"unknown reinforcement model {model_name!r}")
    set_global_seed(config.train.random_seed)
    label = config.train.ranking_target
    df = panel(config, label=label)
    x, y, dates, features, _ = design_matrix(df, label)
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("reinforcement training requires a non-empty feature panel")
    # Oracle is diagnostic-only: it is the realized same-date reward ordering,
    # never an input to the policy.
    requested_top_k = max(1, int(config.constraints.max_positions or 3))
    # LinUCB's panel evaluator requires at least 2*k arms per date so that the
    # selected set is not the entire cross-section. Cap k for narrow universes.
    n_arms = int(np.max([np.sum(np.asarray(dates) == date) for date in set(np.asarray(dates).tolist())]))
    top_k = min(requested_top_k, max(1, n_arms // 2))
    runner = run_linucb_panel if model_name == "linucb" else run_thompson_panel
    trace = runner(
        x,
        y,
        dates,
        oracle=y,
        top_k=top_k,
        alpha=1.0,
        seed=config.train.random_seed,
    )
    if trace.policy_reward.size == 0:
        raise ValueError("reinforcement training produced no evaluable decision dates")
    policy_mean = float(np.mean(trace.policy_reward))
    uniform_mean = float(np.mean(trace.uniform_reward))
    oracle_mean = float(np.mean(trace.oracle_reward))
    metrics = {
        "mean_policy_reward": policy_mean,
        "mean_uniform_reward": uniform_mean,
        "mean_oracle_reward": oracle_mean,
        "mean_advantage_vs_uniform": policy_mean - uniform_mean,
        "mean_regret_vs_oracle": float(np.mean(trace.cumulative_regret)),
        "n_dates": float(len(trace.dates)),
        "n_features": float(x.shape[1]),
    }
    policy = (
        LinUCBRanker(x.shape[1], alpha=1.0)
        if model_name == "linucb"
        else LinearThompsonRanker(x.shape[1], alpha=1.0, seed=config.train.random_seed)
    )
    # Refit the final state causally in the same order used for evaluation.
    for date in sorted(set(np.asarray(dates).tolist())):
        mask = np.asarray(dates) == date
        finite = np.isfinite(y[mask]) & np.isfinite(x[mask]).all(axis=1)
        if not finite.any():
            continue
        scores = policy.scores(x[mask][finite])
        for row in np.argsort(scores)[-top_k:]:
            policy.update(x[mask][finite][row], float(y[mask][finite][row]))
    configure_tracking()
    run_id = log_run(
        family="reinforcement",
        name=model_name,
        params={"features": features, "label": label, "model": model_name},
        metrics=metrics,
        tags={"data": config.data.source, "claim": "research_only"},
    )
    path = Path(config.data.root) / "metadata" / f"rl_{model_name}.joblib"
    save_joblib_artifact(
        {"policy": policy, "policy_name": model_name, "features": features, "label": label},
        path,
    )
    return {
        "metrics": metrics,
        "run_id": run_id,
        "path": str(path),
        "research_only": True,
        "live_pnl_claim": False,
        "data_source": "SYNTHETIC" if config.data.source == "synthetic" else config.data.source,
    }


def train_reinforcement_auto(config: AppConfig) -> dict[str, Any]:
    """Select a contextual policy by chronological research diagnostics."""
    candidates = ("linucb", "thompson", "quantile_thompson", "policy_gradient")
    results: list[dict[str, Any]] = []
    unavailable: dict[str, str] = {}
    for candidate in candidates:
        try:
            results.append(train_reinforcement(config, candidate))
        except ImportError as exc:
            if candidate != "policy_gradient":
                raise
            unavailable[candidate] = str(exc)
    viable = [
        result
        for result in results
        if np.isfinite(float(result["metrics"].get("mean_advantage_vs_uniform", np.nan)))
    ]
    if not viable:
        raise ValueError("automatic RL selection produced no finite candidate metric")
    selected = max(
        viable,
        key=lambda result: float(result["metrics"]["mean_advantage_vs_uniform"]),
    )
    selected_payload = load_joblib_artifact(Path(str(selected["path"])))
    if not isinstance(selected_payload, dict) or "policy" not in selected_payload:
        raise ValueError("selected RL artifact is malformed")
    selected_name = Path(str(selected["path"])).stem.removeprefix("rl_")
    selected_payload["policy_name"] = selected_name
    auto_path = Path(config.data.root) / "metadata" / "rl_auto.joblib"
    save_joblib_artifact(selected_payload, auto_path)
    candidate_diagnostics = {
        Path(str(result["path"])).stem.removeprefix("rl_"): result["metrics"]
        for result in results
    }
    candidate_diagnostics.update(
        {name: {"unavailable": reason} for name, reason in unavailable.items()}
    )
    return {
        **selected,
        "path": str(auto_path),
        "model": "auto",
        "selected_model": selected_name,
        "selection_metric": "mean_advantage_vs_uniform",
        "candidates": candidate_diagnostics,
    }


def train_quantile_bandit(config: AppConfig) -> dict[str, Any]:
    """Train and persist the quantile-Thompson contextual bandit."""
    set_global_seed(config.train.random_seed)
    label = config.train.ranking_target
    df = panel(config, label=label)
    x, y, dates, features, _ = design_matrix(df, label)
    top_k = max(1, int(config.constraints.max_positions or 3))
    n_dates = max(len(set(dates.tolist())), 1)
    arms_per_date = max(1, int(x.shape[0] // n_dates))
    top_k = min(top_k, max(1, arms_per_date // 2))
    bandit = QuantileThompson(seed=config.train.random_seed)
    trace = bandit.run_panel(x, y, dates, k=top_k)
    if not trace.policy_reward.size:
        raise ValueError("quantile bandit training produced no evaluable dates")
    metrics = {
        "mean_policy_reward": float(np.mean(trace.policy_reward)),
        "mean_uniform_reward": float(np.mean(trace.uniform_reward)),
        "mean_oracle_reward": float(np.mean(trace.oracle_reward)),
        "mean_advantage_vs_uniform": float(
            np.mean(trace.policy_reward) - np.mean(trace.uniform_reward)
        ),
        "mean_regret_vs_oracle": float(np.mean(trace.cumulative_regret)),
        "n_dates": float(len(trace.dates)),
    }
    configure_tracking()
    run_id = log_run(
        family="reinforcement",
        name="quantile_thompson",
        params={"features": features, "label": label, "model": "quantile_thompson"},
        metrics=metrics,
        tags={"data": config.data.source, "claim": "research_only"},
    )
    path = Path(config.data.root) / "metadata" / "rl_quantile_thompson.joblib"
    save_joblib_artifact(
        {
            "policy": bandit,
            "policy_name": "quantile_thompson",
            "features": features,
            "label": label,
        },
        path,
    )
    return {
        "metrics": metrics,
        "run_id": run_id,
        "path": str(path),
        "features": features,
        "research_only": True,
        "live_pnl_claim": False,
        "data_source": "SYNTHETIC" if config.data.source == "synthetic" else config.data.source,
    }


def train_policy_gradient(config: AppConfig) -> dict[str, Any]:
    """Train and persist the optional PyTorch contextual policy."""
    set_global_seed(config.train.random_seed)
    label = config.train.ranking_target
    df = panel(config, label=label)
    x, y, dates, features, _ = design_matrix(df, label)
    top_k = max(1, int(config.constraints.max_positions or 3))
    n_dates = max(len(set(dates.tolist())), 1)
    arms_per_date = max(1, int(x.shape[0] // n_dates))
    effective_top_k = min(top_k, max(1, arms_per_date // 2))
    trace = run_policy_gradient_panel(
        x,
        y,
        dates,
        top_k=effective_top_k,
        seed=config.train.random_seed,
    )
    if not trace.policy_reward.size:
        raise ValueError("policy-gradient training produced no evaluable dates")
    metrics = {
        "mean_policy_reward": float(np.mean(trace.policy_reward)),
        "mean_uniform_reward": float(np.mean(trace.uniform_reward)),
        "mean_advantage_vs_uniform": float(
            np.mean(trace.policy_reward) - np.mean(trace.uniform_reward)
        ),
        "mean_regret_vs_uniform": float(np.mean(trace.cumulative_regret)),
        "n_dates": float(len(trace.dates)),
        "n_features": float(x.shape[1]),
    }
    model = PolicyGradientRanker(x.shape[1], seed=config.train.random_seed).fit(
        x, y, dates, top_k=effective_top_k
    )
    configure_tracking()
    run_id = log_run(
        family="reinforcement",
        name="policy_gradient",
        params={"features": features, "label": label, "model": "policy_gradient"},
        metrics=metrics,
        tags={"data": config.data.source, "claim": "research_only"},
    )
    path = Path(config.data.root) / "metadata" / "rl_policy_gradient.joblib"
    save_joblib_artifact(
        {
            "policy": model,
            "policy_name": "policy_gradient",
            "features": features,
            "label": label,
        },
        path,
    )
    return {
        "metrics": metrics,
        "run_id": run_id,
        "path": str(path),
        "features": features,
        "research_only": True,
        "live_pnl_claim": False,
        "data_source": "SYNTHETIC" if config.data.source == "synthetic" else config.data.source,
    }


def train_robinhood_plus(
    config: AppConfig, model_name: str = "hierarchical_markov"
) -> dict[str, Any]:
    """Persist the robinhood+ engine card. The tokenizer is unsupervised."""
    _require_model(model_name, {"hierarchical_markov", "transformer"}, "robinhood_plus")
    from quant_fund.models.robinhood_plus.bench import bench_robinhood_plus
    from quant_fund.models.robinhood_plus.engine import RobinhoodPlusEngine

    engine = RobinhoodPlusEngine(
        lookback=config.robinhood_plus.lookback,
        pred_len=config.robinhood_plus.pred_len,
        sample_count=config.robinhood_plus.sample_count,
        s1_bits=config.robinhood_plus.s1_bits,
        s2_bits=config.robinhood_plus.s2_bits,
        seed=config.train.random_seed,
        decoder=model_name,
    )
    engine.fit(np.zeros((2, 1)), np.zeros(2))
    path = Path(config.data.root) / "metadata" / f"robinhood_plus_{model_name}.joblib"
    engine.save(path)
    df = panel(config)
    receipt = bench_robinhood_plus(df, config)
    return {
        "metrics": {
            "mean_ic": receipt.get("mean_ic"),
            "n_ok": receipt.get("n_ok"),
            "ic_n_dates": receipt.get("ic_n_dates"),
        },
        "path": str(path),
        "family": "robinhood_plus",
        "research_only": True,
        "execution_claim": "research_only",
    }


def train_family(config: AppConfig, family: str, model_name: str | None = None) -> dict[str, Any]:
    dispatch = {
        "ranking": lambda: train_ranking(config, model_name or "ridge"),
        "calibration": lambda: train_calibration(config, model_name or "isotonic"),
        "alpha": lambda: train_alpha(config, model_name or "ridge"),
        "distribution": lambda: train_distribution_auto(config) if model_name == "auto" else train_distribution(config, model_name or "gaussian"),
        "volatility": lambda: train_volatility_auto(config) if model_name == "auto" else train_volatility(config, model_name or "ewma"),
        "regime": lambda: train_regime(config, model_name or "hmm"),
        "tail": lambda: train_tail(config, model_name or "historical"),
        "reinforcement": lambda: train_reinforcement(config, model_name or "linucb"),
        "covariance": lambda: {"metrics": {}, "note": "covariance is estimated at forecast time"},
        "liquidity": lambda: {"metrics": {}, "note": "liquidity uses parameterized cost model"},
        "robinhood_plus": lambda: train_robinhood_plus(config, model_name or "hierarchical_markov"),
    }
    if family not in dispatch:
        raise ValueError(f"unknown family {family}")
    result = dispatch[family]()
    if config.data.source == "synthetic":
        result["data_source"] = "SYNTHETIC"
    artifact_path = result.get("path")
    if isinstance(artifact_path, str):
        try:
            result.update(artifact_identity(Path(artifact_path)))
        except ValueError as exc:
            result["manifest_valid"] = False
            result["artifact_identity_error"] = str(exc)
        try:
            result["dataset_content_sha256"] = canonical_frame_fingerprint(panel(config))
        except (OSError, ValueError, RuntimeError) as exc:
            result["dataset_fingerprint_error"] = str(exc)
    if (
        isinstance(result.get("run_id"), str)
        and result.get("manifest_valid") is True
        and isinstance(result.get("artifact_sha256"), str)
    ):
        attach_artifact_identity(
            str(result["run_id"]),
            {
                "artifact_sha256": result["artifact_sha256"],
                "manifest_valid": result["manifest_valid"],
                "artifact_class": result.get("artifact_class"),
                "manifest_schema": result.get("manifest_schema"),
            },
        )
    report_paths = write_evidence_report(
        Path(config.data.root),
        candidates={family: result.get("metrics", {})},
        provenance={
            "data_source": result.get("data_source", config.data.source),
            "artifact_path": result.get("path"),
            "artifact_sha256": result.get("artifact_sha256"),
            "manifest_valid": result.get("manifest_valid"),
            "config_sha256": hash_bytes(canonical_json_bytes(config.model_dump(mode="json"))),
            "dataset_content_sha256": result.get("dataset_content_sha256"),
            "git_revision": git_revision(),
            "git_worktree_sha256": git_worktree_sha256(),
        },
    )
    result["evidence_report"] = {key: str(path) for key, path in report_paths.items()}
    return result
