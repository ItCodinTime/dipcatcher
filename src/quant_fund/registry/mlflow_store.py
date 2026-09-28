"""Local MLflow registry with candidate/champion/shadow/retired aliases."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from quant_fund.config.models import AppConfig, PromotionConfig
from quant_fund.utils.hashing import fingerprint
from quant_fund.utils.numeric import positive_integral_count

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
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError(f"metric {k} is non-finite")
            mlflow.log_metric(k, float(v))
        mlflow.set_tags(tags)
        if artifact_dir is not None and Path(artifact_dir).exists():
            mlflow.log_artifacts(str(artifact_dir))
        return str(run.info.run_id)


def set_alias(run_id: str, alias: str, *, promotion: dict[str, Any] | None = None) -> None:
    """Assign a registry alias, requiring a passing receipt for champion.

    Candidate/shadow/retired are reversible workflow states. Champion is a
    production-facing claim and therefore cannot be assigned by a bare tag.
    """
    if alias not in ALIASES:
        raise ValueError(f"alias must be one of {ALIASES}")
    if alias == "champion" and not promotion_is_approved(promotion, run_id=run_id):
        raise PermissionError("champion alias requires an approved promotion receipt")
    client = MlflowClient()
    run = client.get_run(run_id)
    client.set_tag(run_id, "alias", alias)
    client.set_tag(run_id, f"alias_{alias}", "true")
    _ = run


def attach_artifact_identity(run_id: str, identity: dict[str, Any]) -> None:
    """Attach verified artifact identity to an existing MLflow run."""
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be non-empty")
    digest = identity.get("artifact_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("artifact identity requires a 64-character sha256")
    int(digest, 16)
    if identity.get("manifest_valid") is not True:
        raise ValueError("artifact identity manifest must be valid")
    client = MlflowClient()
    for key, value in {
        "artifact_sha256": digest,
        "artifact_manifest_valid": "true",
        "artifact_class": str(identity.get("artifact_class", "")),
        "artifact_manifest_schema": str(identity.get("manifest_schema", "")),
    }.items():
        client.set_tag(run_id, key, value)


def promotion_is_approved(promotion: dict[str, Any] | None, *, run_id: str | None = None) -> bool:
    """Validate the explicit promotion receipt required for champion status.

    Fail closed: missing identity/evidence, mismatched run_id, or synthetic
    research artifacts never authorize champion.  The schema marker and
    leakage flag prevent a caller from mistaking a hand-crafted metrics blob
    for the output of :func:`promotion_decision`.
    """
    if not isinstance(promotion, dict):
        return False
    # Synthetic receipts are never champion — even if tampered to promote=True.
    if bool(promotion.get("synthetic", False)):
        return False
    data_source = promotion.get("data_source")
    normalized_source = data_source.strip().upper() if isinstance(data_source, str) else ""
    if normalized_source == "SYNTHETIC":
        return False
    if "synthetic_evidence_not_promotable" in list(promotion.get("reasons") or []):
        return False
    artifact_sha256 = promotion.get("artifact_sha256")
    if artifact_sha256 is not None:
        if not isinstance(artifact_sha256, str) or len(artifact_sha256) != 64:
            return False
        try:
            int(artifact_sha256, 16)
        except ValueError:
            return False
        if promotion.get("manifest_valid") is not True:
            return False
    return bool(
        promotion.get("receipt_schema") == "promotion.v1"
        and promotion.get("promote") is True
        and promotion.get("evidence_complete") is True
        and promotion.get("research_receipt_valid") is True
        and promotion.get("leakage_ok") is True
        and isinstance(data_source, str)
        and bool(data_source.strip())
        and promotion.get("reasons") == []
        and isinstance(promotion.get("run_id"), str)
        and bool(promotion.get("run_id"))
        and (run_id is None or promotion.get("run_id") == run_id)
    )


def promotion_decision(
    metrics: dict[str, Any], cfg: PromotionConfig, leakage_ok: bool
) -> dict[str, Any]:
    """Return a fail-closed promotion decision.

    Promotion is an evidence decision, not a threshold lookup. Missing or
    non-finite evidence must never be interpreted as zero, and synthetic
    research artifacts are never eligible for a production alias.
    """
    reasons: list[str] = []
    ok = True
    raw_data_source = metrics.get("data_source")
    data_source = raw_data_source.strip() if isinstance(raw_data_source, str) else ""
    raw_synthetic = metrics.get("synthetic", False)
    synthetic_flag = raw_synthetic is True
    if "synthetic" in metrics and type(raw_synthetic) is not bool:
        ok = False
        reasons.append("synthetic_flag_invalid")
    if not data_source:
        ok = False
        reasons.append("data_source_missing")
    if synthetic_flag or data_source.upper() == "SYNTHETIC":
        ok = False
        reasons.append("synthetic_evidence_not_promotable")
    if metrics.get("evidence_complete") is not True:
        ok = False
        reasons.append("evidence_incomplete")
    artifact_sha256 = metrics.get("artifact_sha256")
    if artifact_sha256 is not None:
        if not isinstance(artifact_sha256, str) or len(artifact_sha256) != 64:
            ok = False
            reasons.append("artifact_hash_invalid")
        else:
            try:
                int(artifact_sha256, 16)
            except ValueError:
                ok = False
                reasons.append("artifact_hash_invalid")
    if metrics.get("manifest_valid") is False:
        ok = False
        reasons.append("artifact_manifest_invalid")
    leakage_pass = leakage_ok is True
    if cfg.require_leakage_pass and not leakage_pass:
        ok = False
        reasons.append("leakage_failed")
    required = ("mean_ic", "net_spread", "turnover")
    missing = [k for k in required if k not in metrics or not _finite_number(metrics[k])]
    if missing:
        ok = False
        reasons.append("missing_metrics:" + ",".join(missing))
    mean_ic = _numeric_or_nan(metrics.get("mean_ic"))
    if mean_ic < cfg.min_mean_ic:
        ok = False
        reasons.append("ic_below_min")
    spread = _numeric_or_nan(metrics.get("net_spread"))
    if spread < cfg.min_cost_adjusted_spread:
        ok = False
        reasons.append("spread_below_min")
    to = _numeric_or_nan(metrics.get("turnover"))
    if to > cfg.max_turnover:
        ok = False
        reasons.append("turnover_high")
    # Multi-fold stability evidence (fail closed when required folds missing)
    n_folds = metrics.get("n_folds")
    if n_folds is None:
        n_folds = metrics.get("n_ic_folds")
    n_folds_i = _positive_integral_count(n_folds)
    if n_folds_i < int(cfg.min_folds):
        ok = False
        reasons.append("insufficient_folds")
    fold_stab = metrics.get("fold_ic_stability")
    if fold_stab is not None:
        fs = float(fold_stab) if _finite_number(fold_stab) else float("nan")
        if not math.isfinite(fs) or fs < float(cfg.min_fold_ic_stability):
            ok = False
            reasons.append("fold_ic_unstable")
    return {
        "receipt_schema": "promotion.v1",
        "promote": ok,
        "reasons": reasons,
        "levels_visible": True,
        "evidence_complete": metrics.get("evidence_complete") is True,
        # Fail closed: absent receipt-validity evidence is never True. Callers
        # that ran the verifier patch this field explicitly (see gates.py).
        "research_receipt_valid": bool(metrics.get("research_receipt_valid", False)),
        "leakage_ok": leakage_pass,
        "data_source": data_source,
        "synthetic": synthetic_flag,
        "run_id": metrics.get("run_id"),
        "artifact_sha256": artifact_sha256,
        "manifest_valid": metrics.get("manifest_valid"),
    }


def _finite_number(value: Any) -> bool:
    """Accept numeric scalar evidence, excluding bools and NaN/inf."""
    import math

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _numeric_or_nan(value: Any) -> float:
    """Normalize untrusted metric payloads without raising during gating."""
    return float(value) if _finite_number(value) else float("nan")


def _positive_integral_count(value: object) -> int:
    """Parse fold counts without accepting truthy or string coercions."""
    return positive_integral_count(value)


def log_fleet_run(receipt_path: Path, receipt: dict[str, Any], *, uri: str | None = None) -> str:
    """Index a fleet tournament run in MLflow, keyed to its sealed receipt.

    MLflow is the discoverability index, not the evidence: the sealed
    ``receipts/*.json`` file is canonical and immutable, while run rows are
    mutable local state. Every logged fleet run therefore carries the
    receipt's sha256 so an index row can be traced to — and audited against —
    the exact evidence bytes. Aggregate metrics are proper scores only
    (CRPS / PIT-KS), aggregated over ``status == "ok"`` result rows. Fails
    closed on a non-fleet receipt, a missing file, or a tournament with no
    scored rows.
    """
    path = Path(receipt_path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"receipt file missing or not a regular file: {path}")
    import hashlib

    receipt_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    data_label = receipt.get("data_label")
    if not isinstance(data_label, str) or not data_label.strip():
        raise ValueError("receipt is missing a data_label")
    results = receipt.get("results")
    if not isinstance(results, list):
        raise ValueError("receipt is missing a results list")
    ok_rows = [r for r in results if isinstance(r, dict) and r.get("status") == "ok"]
    if not ok_rows:
        raise ValueError("fleet receipt has no successfully scored rows")

    def _col(key: str) -> list[float]:
        return [
            float(r[key])
            for r in ok_rows
            if isinstance(r.get(key), (int, float))
            and not isinstance(r.get(key), bool)
            and math.isfinite(float(r[key]))
        ]

    crps = _col("crps")
    pit_ks = _col("pit_ks")
    per_head: dict[str, list[float]] = {}
    for r in ok_rows:
        if isinstance(r.get("crps"), (int, float)) and math.isfinite(float(r["crps"])):
            per_head.setdefault(str(r.get("model")), []).append(float(r["crps"]))
    head_means = {m: math.fsum(v) / len(v) for m, v in per_head.items()}

    models = receipt.get("models") or []
    shards = receipt.get("shards") or {}
    metrics: dict[str, float] = {
        "n_rows": float(len(results)),
        "n_ok_rows": float(len(ok_rows)),
        "n_error_rows": float(len(results) - len(ok_rows)),
        "n_heads": float(len(models)) if models else float(len(per_head)),
        "n_shards": float(len(shards)),
    }
    if crps:
        metrics["mean_crps"] = math.fsum(crps) / len(crps)
    if pit_ks:
        metrics["mean_pit_ks"] = math.fsum(pit_ks) / len(pit_ks)
    if head_means:
        metrics["best_head_crps"] = min(head_means.values())
        metrics["worst_head_crps"] = max(head_means.values())

    configure_tracking(uri)
    return log_run(
        family="fleet_tournament",
        name=f"fleet-{receipt_sha256[:12]}",
        params={
            "seed": receipt.get("seed"),
            "n_train": receipt.get("n_train"),
            "n_eval": receipt.get("n_eval"),
            "taus": receipt.get("taus"),
            "models": ",".join(str(m) for m in models),
            "shards": ",".join(str(s) for s in shards),
            "receipt_schema": receipt.get("schema"),
            "receipt_path": str(path),
        },
        metrics=metrics,
        tags={
            "data": data_label,
            "kind": str(receipt.get("kind", "")),
            "evidence": "receipt",
            "receipt_sha256": receipt_sha256,
        },
    )


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
