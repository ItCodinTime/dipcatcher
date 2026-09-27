"""Ranking benches or ranking and calibration training.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.lightspeed.ranker import NauticaRanker
from quant_fund.metrics.cross_section import date_ic_series
from quant_fund.metrics.probability import brier_score
from quant_fund.metrics.scoring import icir
from quant_fund.models.asset_pricing import IPCARanker, RandomFourierRanker, SDFRidgeRanker
from quant_fund.models.base import JoblibMixin
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
from quant_fund.pipeline.dataset import design_matrix, panel
from quant_fund.registry.mlflow_store import configure_tracking, log_run
from quant_fund.utils.seeds import set_global_seed

from .splits import _aligned_label_end_times, _label_horizon, _require_model, _walk_forward_splits

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
        result for result in results if np.isfinite(float(result["metrics"].get("mean_ic", np.nan)))
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
            [
                dates[test_mask]
                for _train_mask, test_mask in _walk_forward_splits(
                    dates, config, horizon_bars=_label_horizon(label)
                )
                if test_mask.any()
            ]
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


__all__ = [
    "RANKING_MODEL_NAMES",
    "train_calibration",
    "train_calibration_auto",
    "train_ranking",
    "train_ranking_auto",
]
