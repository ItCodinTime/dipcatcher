"""Distribution-family training.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.config.models import AppConfig
from quant_fund.metrics.scoring import mean_pinball, quantile_crossing_rate
from quant_fund.models.base import load_joblib_artifact, save_joblib_artifact
from quant_fund.models.conformal_dist import ConformalTDistribution
from quant_fund.models.distribution import (
    EmpiricalDistribution,
    GaussianDistribution,
    GMMDistribution,
    IsotonicPitDistribution,
    LinearQuantileDistribution,
    SkewTDistribution,
    StackedDistribution,
    TreeQuantileDistribution,
)
from quant_fund.models.fhs import FhsSkewDistribution
from quant_fund.models.lgbm_q2 import LGBMQ2Distribution
from quant_fund.models.regime_dist import RegimeDistribution
from quant_fund.pipeline.dataset import design_matrix, panel
from quant_fund.registry.mlflow_store import configure_tracking, log_run
from quant_fund.utils.seeds import set_global_seed

from .splits import _aligned_label_end_times, _label_horizon, _require_model, _walk_forward_splits


def train_distribution(config: AppConfig, model_name: str = "gaussian") -> dict[str, Any]:
    _require_model(
        model_name,
        {
            "empirical",
            "gaussian",
            "linear_qr",
            "xgboost",
            "lightgbm",
            "lgbm_q2",
            "skew_t",
            "gmm",
            "isotonic",
            "stack",
            "conf_t",
            "fhs_skew",
            "regime",
        },
        "distribution",
    )
    set_global_seed(config.train.random_seed)
    label = config.train.distribution_target
    df = panel(config, label=label)
    x, y, dates, feats, ids = design_matrix(df, label)
    if model_name in {"fhs_skew", "regime"} and (
        ids.size == 0
        or np.unique(ids).size != 1
        or np.unique(dates).size != dates.size
        or (dates.size > 1 and not np.all(dates[1:] > dates[:-1]))
    ):
        raise ValueError(f"{model_name} requires one security with strictly increasing event times")
    label_end_times = _aligned_label_end_times(df, label, feats)
    taus = config.quantiles.levels

    def make_model() -> Any:
        catalog: dict[str, Callable[[], Any]] = {
            "empirical": lambda: EmpiricalDistribution(taus),
            "gaussian": lambda: GaussianDistribution(taus),
            "linear_qr": lambda: LinearQuantileDistribution(taus),
            "xgboost": lambda: TreeQuantileDistribution(taus, "xgboost", config.train.random_seed),
            "lightgbm": lambda: TreeQuantileDistribution(
                taus, "lightgbm", config.train.random_seed
            ),
            "lgbm_q2": lambda: LGBMQ2Distribution(taus, seed=config.train.random_seed),
            "skew_t": lambda: SkewTDistribution(taus),
            "gmm": lambda: GMMDistribution(taus, seed=config.train.random_seed),
            "isotonic": lambda: IsotonicPitDistribution(taus),
            "stack": lambda: StackedDistribution(taus, seed=config.train.random_seed),
            "conf_t": lambda: ConformalTDistribution(taus),
            "fhs_skew": lambda: FhsSkewDistribution(taus),
            "regime": lambda: RegimeDistribution(taus, seed=config.train.random_seed),
        }
        if model_name not in catalog:
            raise ValueError(f"unknown distribution model {model_name!r}")
        return catalog[model_name]()

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
        "candidates": {
            Path(str(r["path"])).stem.removeprefix("dist_"): r["metrics"] for r in results
        },
    }


__all__ = [
    "train_distribution",
    "train_distribution_auto",
]
