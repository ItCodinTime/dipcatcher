"""Local MLflow registry with candidate/champion/shadow/retired aliases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from quant_fund.config.models import AppConfig, PromotionConfig
from quant_fund.utils.hashing import fingerprint

ALIASES = ("candidate", "champion", "shadow", "retired")


def configure_tracking(uri: str | None = None) -> None:
    import os

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    # MLflow 3.x file store is in maintenance; default to local sqlite.
    mlflow.set_tracking_uri(uri or "sqlite:///mlflow.db")


def log_run(
    *,
    family: str,
    name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, str],
    artifact_dir: Path | None = None,
) -> str:
    mlflow.set_experiment(family)
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        for k, v in metrics.items():
            if v is None or (isinstance(v, float) and (v != v)):  # nan
                continue
            mlflow.log_metric(k, float(v))
        mlflow.set_tags(tags)
        if artifact_dir is not None and Path(artifact_dir).exists():
            mlflow.log_artifacts(str(artifact_dir))
        return run.info.run_id


def set_alias(run_id: str, alias: str) -> None:
    if alias not in ALIASES:
        raise ValueError(f"alias must be one of {ALIASES}")
    client = MlflowClient()
    run = client.get_run(run_id)
    client.set_tag(run_id, "alias", alias)
    client.set_tag(run_id, f"alias_{alias}", "true")
    _ = run


def promotion_decision(
    metrics: dict[str, float], cfg: PromotionConfig, leakage_ok: bool
) -> dict[str, Any]:
    reasons: list[str] = []
    ok = True
    if cfg.require_leakage_pass and not leakage_ok:
        ok = False
        reasons.append("leakage_failed")
    mean_ic = metrics.get("mean_ic", 0.0)
    if mean_ic < cfg.min_mean_ic:
        ok = False
        reasons.append("ic_below_min")
    spread = metrics.get("net_spread", 0.0)
    if spread < cfg.min_cost_adjusted_spread:
        ok = False
        reasons.append("spread_below_min")
    to = metrics.get("turnover", 0.0)
    if to > cfg.max_turnover:
        ok = False
        reasons.append("turnover_high")
    return {"promote": ok, "reasons": reasons, "levels_visible": True}


def dataset_fingerprint_from_frame(
    n: int, cols: list[str], tmin: str, tmax: str, config: AppConfig
) -> str:
    return fingerprint(
        row_count=n,
        min_timestamp=tmin,
        max_timestamp=tmax,
        columns=cols,
        feature_version="features.v1",
        universe_version="universe.v1",
        label_version="labels.v1",
        extra={"seed": config.train.random_seed},
    )
