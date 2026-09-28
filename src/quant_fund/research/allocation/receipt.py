"""Lightweight evaluation receipts for allocation runs.

This is *not* the sealed research-receipt machinery
(``quant_fund.research.catalog`` / ``verify_research_artifact``) — those
receipts gate forecaster notebooks and are immutable, so a new evaluation
pack must not retrofit into them. Instead this module emits a
self-contained, hash-bound JSON document reusing the same primitives
(``canonical_json_bytes`` / ``receipt_tree`` / ``hash_bytes``) plus the
catalog's fail-closed forbidden-headline-key check
(``family_blob_forbidden_metrics_absent``), which runs on every metric
block before a receipt is written.

The payload hash is computed over the receipt minus the ``payload_sha256``
field; ``verify_allocation_receipt`` re-derives it from disk. The receipt
records calibration/stability metrics only — a ``synthetic`` flag marks
synthetic-input runs and is required to be explicit (never defaults to
real).
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_fund.research.allocation.evaluation import AllocationEvaluation
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes, receipt_tree

ALLOCATION_RECEIPT_SCHEMA = "allocation_evaluation.v1"


def _atomic_write(path: Path, payload: bytes) -> None:
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def _weights_digest(evaluation: AllocationEvaluation) -> str:
    canonical = receipt_tree(evaluation.weights.tolist())
    return hash_bytes(canonical_json_bytes(canonical))


def build_receipt(
    evaluation: AllocationEvaluation,
    *,
    synthetic: bool,
    label: str | None = None,
    parameters: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a hash-bound receipt dict for one walk-forward evaluation.

    ``synthetic`` must be passed explicitly — it marks whether the return
    series was synthetic, so the honesty label travels with the evidence.
    Raises ``ValueError`` if the metrics block contains a forbidden
    headline key (checked with the same helper the research catalog uses).
    """
    metrics = receipt_tree(dict(evaluation.metrics))
    if not family_blob_forbidden_metrics_absent(metrics):
        raise ValueError("evaluation metrics contain a forbidden headline key")
    receipt: dict[str, Any] = {
        "schema": ALLOCATION_RECEIPT_SCHEMA,
        "claim": "research_only",
        "synthetic": bool(synthetic),
        "generated_at": generated_at or datetime.now(UTC).isoformat(),
        "engine": evaluation.engine,
        "parameters": receipt_tree(dict(parameters or {})),
        "data": {
            "n_rebalances": int(evaluation.decision_index.size),
            "n_assets": int(evaluation.n_assets),
            "window": int(evaluation.window),
            "step": int(evaluation.step),
            "horizon": int(evaluation.horizon),
            "first_decision_index": int(evaluation.decision_index[0])
            if evaluation.decision_index.size
            else None,
            "last_decision_index": int(evaluation.decision_index[-1])
            if evaluation.decision_index.size
            else None,
            "weights_sha256": _weights_digest(evaluation),
        },
        "metrics": metrics,
        "constraint_violations": int(evaluation.constraint_violations),
    }
    if label is not None:
        receipt["label"] = str(label)
    payload = {k: v for k, v in receipt.items() if k != "payload_sha256"}
    receipt["payload_sha256"] = hash_bytes(canonical_json_bytes(receipt_tree(payload)))
    return receipt


def write_receipt(receipt: dict[str, Any], path: Path | str) -> Path:
    """Write a receipt JSON atomically. Rejects non-v1 payloads."""
    if not isinstance(receipt, dict) or receipt.get("schema") != ALLOCATION_RECEIPT_SCHEMA:
        raise ValueError(f"receipt must be a {ALLOCATION_RECEIPT_SCHEMA} payload")
    if not family_blob_forbidden_metrics_absent(receipt):
        raise ValueError("receipt contains a forbidden headline key")
    destination = Path(path)
    body = json.dumps(receipt_tree(receipt), indent=2, sort_keys=True) + "\n"
    _atomic_write(destination, body.encode("utf-8"))
    return destination


def verify_allocation_receipt(path: Path | str) -> dict[str, Any]:
    """Re-derive the payload hash of a receipt on disk.

    Returns ``{"valid": bool, "errors": [...]}`` — additive check, exactly
    like the explainability sidecar verifier: it proves the receipt file's
    own integrity, nothing more.
    """
    errors: list[str] = []
    p = Path(path)
    try:
        payload = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "errors": [f"receipt_unreadable:{exc}"]}
    if not isinstance(payload, dict) or payload.get("schema") != ALLOCATION_RECEIPT_SCHEMA:
        errors.append("receipt_schema_mismatch")
        payload = {}
    if payload.get("claim") != "research_only":
        errors.append("receipt_claim_mismatch")
    if "synthetic" not in payload or not isinstance(payload.get("synthetic"), bool):
        errors.append("receipt_synthetic_missing")
    if payload and not family_blob_forbidden_metrics_absent(payload):
        errors.append("receipt_forbidden_metrics")
    expected = payload.get("payload_sha256")
    unsigned = {k: v for k, v in payload.items() if k != "payload_sha256"}
    actual = hash_bytes(canonical_json_bytes(receipt_tree(unsigned)))
    if expected != actual:
        errors.append("receipt_payload_hash_mismatch")
    return {"valid": not errors, "errors": errors}
