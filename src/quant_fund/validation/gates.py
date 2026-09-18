"""Fail-closed research / promotion validation gates.

``quant validate`` is an evidence decision, not a green-stamp. SYNTHETIC runs
may pass research-correctness checks but can never promote to a live alias.
Missing causal panels or incomplete walk-forward evidence fail closed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.registry.mlflow_store import promotion_decision
from quant_fund.research.verify import verify_research_artifact
from quant_fund.utils.reproducibility import git_worktree_sha256

REQUIRED_EVIDENCE = ("mean_ic", "net_spread", "turnover")


def _positive_integral_count(value: Any) -> int:
    """Return a valid positive count, rejecting coercion-shaped evidence."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
        return 0
    return int(numeric)


def _required_metrics_present(metrics: dict[str, Any]) -> bool:
    """Require finite core evidence rather than trusting a completion marker."""
    return all(
        key in metrics
        and isinstance(metrics[key], (int, float))
        and not isinstance(metrics[key], bool)
        and math.isfinite(float(metrics[key]))
        for key in REQUIRED_EVIDENCE
    )


def _current_worktree_sha256() -> str:
    """Return the same checkout fingerprint used when minting receipts."""
    return git_worktree_sha256()


def _is_synthetic(config: AppConfig, metrics: dict[str, Any]) -> bool:
    if str(config.data.source).lower() == "synthetic":
        return True
    if bool(metrics.get("synthetic", False)):
        return True
    return str(metrics.get("data_source", "")).upper() == "SYNTHETIC"


def _load_metrics(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.is_file():
        return {"_metrics_file_missing": True, "path": str(path)}
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"_metrics_file_invalid": True, "path": str(path)}
    if not isinstance(raw, dict):
        return {"_metrics_file_invalid": True, "path": str(path)}
    return raw


def _load_research_notebook(config: AppConfig) -> dict[str, Any] | None:
    path = Path(config.data.root) / "metadata" / "research" / "latest.json"
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"_notebook_invalid": True}
    return raw if isinstance(raw, dict) else {"_notebook_invalid": True}


def _causal_panel_present(config: AppConfig) -> bool:
    """Validate a real target-weight panel, not just a filename marker."""
    path = Path(config.data.root) / "gold" / "target_weights.parquet"
    if not path.is_file():
        return False
    try:
        frame = pl.read_parquet(path)
    except (OSError, pl.exceptions.PolarsError):
        return False
    required = {"event_time", "security_id", "target_weight"}
    if not required.issubset(frame.columns) or frame.height == 0:
        return False
    valid_rows = frame.select(
        (
            pl.col("event_time").is_not_null()
            & pl.col("security_id").is_not_null()
            & (pl.col("security_id").cast(pl.String).str.len_chars() > 0)
            & pl.col("target_weight").is_finite()
        ).all()
    ).item()
    if not bool(valid_rows):
        return False
    # Duplicate as-of/security rows make the panel ambiguous and can silently
    # double-count exposure in downstream portfolio construction.
    unique_keys = frame.select(pl.struct(["event_time", "security_id"]).n_unique()).item()
    return int(unique_keys) == frame.height


def _walk_forward_evidence(metrics: dict[str, Any], notebook: dict[str, Any] | None) -> bool:
    # A completion marker alone is not evidence: zero, negative, fractional,
    # and non-numeric counts must not satisfy the gate.
    for key in ("n_ic_dates", "n_folds"):
        if _positive_integral_count(metrics.get(key)) > 0:
            return True
    if notebook is None:
        return False
    rankers = notebook.get("rankers") or []
    if isinstance(rankers, list):
        for ranker in rankers:
            if not isinstance(ranker, dict) or not str(ranker.get("name", "")).strip():
                continue
            has_metric = any(
                isinstance(ranker.get(key), (int, float))
                and not isinstance(ranker.get(key), bool)
                and math.isfinite(float(ranker[key]))
                for key in ("mean_ic", "p_ic", "fold_ic_stability")
            )
            if not has_metric:
                continue
            for key in ("n_dates", "n_folds"):
                if _positive_integral_count(ranker.get(key)) > 0:
                    return True
    # Hypothesis rows describe statistical tests; they are not temporal
    # walk-forward evidence and must never satisfy this gate by themselves.
    return False


def _multi_fold_stability(
    metrics: dict[str, Any], notebook: dict[str, Any] | None, min_folds: int
) -> tuple[bool, str | None]:
    """Require multi-fold evidence when promotion-grade stability is claimed."""
    n = metrics.get("n_folds") or metrics.get("n_ic_folds")
    n_i = _positive_integral_count(n)
    if n_i >= min_folds:
        return True, None
    # Infer from notebook ranker date counts if present
    if notebook is not None:
        rankers = notebook.get("rankers") or []
        for r in rankers:
            if not isinstance(r, dict):
                continue
            # Date count demonstrates temporal coverage, but it is not a fold
            # count and must not be substituted for independent validation
            # splits in the promotion-grade stability gate.
            for key in ("n_folds",):
                if _positive_integral_count(r.get(key)) >= min_folds:
                    return True, None
    # A completion marker without a count is not multi-fold evidence. Keep the
    # gate strict even when another gate has accepted a causal panel.
    if n_i > 0 and n_i < min_folds:
        return False, "insufficient_multi_fold_stability"
    return n_i >= min_folds, None


def validate_candidate(
    model_id: str,
    config: AppConfig,
    *,
    metrics: dict[str, Any] | None = None,
    metrics_path: Path | None = None,
    claim_live: bool = False,
    leakage_ok: bool = True,
) -> dict[str, Any]:
    """Run fail-closed validation / promotion gates for ``model_id``.

    Returns a structured report. ``ok`` is True only when research-correctness
    gates pass. ``promote`` is True only when a non-synthetic candidate clears
    ``promotion_decision``. Claiming live on SYNTHETIC always sets ``ok=False``.
    """
    blob = dict(metrics or {})
    loaded = _load_metrics(metrics_path)
    if loaded.get("_metrics_file_missing") or loaded.get("_metrics_file_invalid"):
        reason = (
            "metrics_file_missing"
            if loaded.get("_metrics_file_missing")
            else "metrics_file_invalid"
        )
        return {
            "model_id": model_id,
            "ok": False,
            "promote": False,
            "data_label": "UNKNOWN",
            "reasons": [f"{reason}:{loaded.get('path')}"],
            "gates": {},
        }
    blob.update({k: v for k, v in loaded.items() if not str(k).startswith("_")})

    notebook = _load_research_notebook(config)
    notebook_invalid = bool(notebook and notebook.get("_notebook_invalid"))
    # Promotion evidence must be explicitly bound to a verified immutable
    # receipt. Missing receipts are not equivalent to an optional notebook.
    notebook_receipt_valid = False
    notebook_path = Path(config.data.root) / "metadata" / "research" / "latest.json"
    if notebook is not None and not notebook_invalid:
        receipt = verify_research_artifact(notebook_path)
        notebook_receipt_valid = bool(receipt.get("valid", False))
        if not notebook_receipt_valid:
            notebook_invalid = True
            notebook = None
    if notebook_invalid:
        notebook = None
    synthetic = _is_synthetic(config, blob)
    if notebook is not None and notebook.get("synthetic") is True:
        synthetic = True

    reasons: list[str] = []
    gates: dict[str, bool] = {}
    if notebook_invalid:
        reasons.append("research_notebook_invalid")
    elif notebook is None:
        reasons.append("research_notebook_missing")
    gates["research_notebook_receipt_valid"] = notebook_receipt_valid

    receipt_provenance = notebook.get("provenance") if isinstance(notebook, dict) else None
    receipt_run_id = (
        receipt_provenance.get("run_id") if isinstance(receipt_provenance, dict) else None
    )
    candidate_run_id = blob.get("run_id")
    receipt_run_id_bound = bool(
        notebook_receipt_valid
        and isinstance(receipt_run_id, str)
        and isinstance(candidate_run_id, str)
        and candidate_run_id == receipt_run_id
    )
    gates["research_receipt_run_id_bound"] = receipt_run_id_bound
    if notebook_receipt_valid and not receipt_run_id_bound:
        reasons.append("research_receipt_run_id_unbound")

    receipt_worktree_hash = (
        receipt_provenance.get("git_worktree_sha256")
        if isinstance(receipt_provenance, dict)
        else None
    )
    current_worktree_hash = (
        _current_worktree_sha256() if notebook_receipt_valid and receipt_worktree_hash else None
    )
    receipt_worktree_bound = bool(
        notebook_receipt_valid
        and isinstance(receipt_worktree_hash, str)
        and isinstance(current_worktree_hash, str)
        and current_worktree_hash != "UNKNOWN"
        and current_worktree_hash == receipt_worktree_hash
    )
    gates["research_receipt_worktree_bound"] = receipt_worktree_bound
    if notebook_receipt_valid and not receipt_worktree_bound:
        reasons.append("research_receipt_worktree_unbound")

    # Gate: causal / walk-forward evidence
    # A metrics marker is not causal evidence. The panel must be present at the
    # configured path and pass the structural validator above; otherwise a
    # forged ``causal_panel=true`` field can bypass the evidence boundary.
    causal = _causal_panel_present(config)
    wf = _walk_forward_evidence(blob, notebook)
    gates["causal_or_walk_forward"] = causal or wf
    if not gates["causal_or_walk_forward"]:
        reasons.append("missing_causal_panel_or_walk_forward_evidence")

    fold_ok, fold_reason = _multi_fold_stability(blob, notebook, int(config.promotion.min_folds))
    gates["multi_fold_stability"] = bool(fold_ok)
    if fold_reason and fold_reason not in reasons:
        reasons.append(fold_reason)
    if not fold_ok:
        reasons.append("insufficient_multi_fold_stability")

    # Gate: research notebook family split present when notebook exists
    if notebook is not None:
        hyps = notebook.get("hypotheses") or []
        families = {
            (h.get("family") if isinstance(h, dict) else getattr(h, "family", None)) for h in hyps
        }
        families.discard(None)
        has_split = (
            ("calibration" in families) or ("discovery" in families) or ("bound" in families)
        )
        gates["hypothesis_family_split"] = bool(hyps) and has_split
        if hyps and not has_split:
            reasons.append("hypothesis_family_split_missing")
    else:
        gates["hypothesis_family_split"] = False
        # Notebook optional when explicit metrics are complete
        if blob.get("evidence_complete") is not True:
            reasons.append("research_notebook_missing")

    # Gate: SYNTHETIC cannot be claimed as live
    gates["synthetic_not_claimed_live"] = not (synthetic and claim_live)
    if synthetic and claim_live:
        reasons.append("synthetic_claimed_as_live")

    # Promotion decision (fail-closed)
    promo_metrics = dict(blob)
    if synthetic:
        promo_metrics["data_source"] = "SYNTHETIC"
    if "evidence_complete" not in promo_metrics:
        # Auto-complete only when all required finite metrics are present.
        from quant_fund.registry.mlflow_store import _finite_number

        promo_metrics["evidence_complete"] = all(
            k in promo_metrics and _finite_number(promo_metrics[k]) for k in REQUIRED_EVIDENCE
        )
    promo = promotion_decision(promo_metrics, config.promotion, leakage_ok=leakage_ok)
    promo["research_receipt_valid"] = notebook_receipt_valid
    if not notebook_receipt_valid or not receipt_run_id_bound or not receipt_worktree_bound:
        promo["promote"] = False
        if not notebook_receipt_valid and "research_notebook_invalid" not in reasons:
            reasons.append("research_notebook_invalid")
    gates["promotion"] = bool(promo["promote"])
    for r in promo.get("reasons") or []:
        if r not in reasons:
            reasons.append(str(r))

    # Research-ok: causal/WF present, no synthetic-as-live mistake
    research_ok = (
        gates["causal_or_walk_forward"]
        and gates["synthetic_not_claimed_live"]
        and gates.get("multi_fold_stability", True)
    )
    if notebook is None and not _required_metrics_present(blob):
        research_ok = False

    data_label = "SYNTHETIC" if synthetic else str(config.data.source)
    return {
        "model_id": model_id,
        "ok": bool(research_ok),
        "promote": bool(promo["promote"]),
        "data_label": data_label,
        "synthetic": synthetic,
        "reasons": reasons,
        "gates": gates,
        "promotion": promo,
        "note": (
            "SYNTHETIC evidence is for lab correctness only; never a live-P&L claim."
            if synthetic
            else "Public/file-feature validation; promotion still requires full evidence."
        ),
    }
