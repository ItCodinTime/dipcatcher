"""Receipt sealing, writing, and verification for replication runs.

Mirrors the ``research.fleet_eval`` conventions: the receipt payload is
canonicalized, ``receipt_sha256`` = sha256 of the canonical payload without the
digest field, and the file is written atomically under ``receipts/`` (never
overwritten with different content). ``verify_replication_receipt`` re-checks
the seal plus the honesty invariants so any committed receipt can be
re-validated standalone:

    uv run --no-sync python -m quant_fund.research.replication.verify \
        receipts/paper_replication_<hash>.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.research.fleet_eval import _atomic_write_text
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

REPLICATION_SCHEMA = "paper_replication.v1"
REQUIRED_TOP_KEYS = (
    "schema",
    "kind",
    "data_label",
    "live_pnl_claim",
    "research_only",
    "generated_at",
    "git_revision",
    "data",
    "inputs_sha256",
    "strategies",
    "citations",
    "limitations",
    "disclaimer",
)


def seal_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach ``receipt_sha256`` over the canonical payload (digest-excluded)."""
    canonical = json.loads(canonical_json_bytes(payload))
    digest = hash_bytes(canonical_json_bytes(canonical))
    return {**canonical, "receipt_sha256": digest}


def write_receipt(receipt: dict[str, Any], receipts_dir: Path | str = Path("receipts")) -> Path:
    """Validate honesty invariants, seal, and atomically write the receipt."""
    if receipt.get("schema") != REPLICATION_SCHEMA:
        raise ValueError("receipt schema mismatch")
    if receipt.get("live_pnl_claim") is not False:
        raise ValueError("live_pnl_claim must be false")
    if receipt.get("research_only") is not True:
        raise ValueError("research_only must be true")
    if receipt.get("data_label") != "SYNTHETIC":
        raise ValueError("data_label must be SYNTHETIC (fixtures or synthetic panel)")
    strategies = receipt.get("strategies")
    if not isinstance(strategies, dict) or not strategies:
        raise ValueError("receipt has no strategies")
    research_blob = {k: v for k, v in receipt.items() if k != "live_pnl_claim"}
    if not family_blob_forbidden_metrics_absent(research_blob):
        raise ValueError("receipt contains forbidden research metric keys")
    sealed = seal_receipt(receipt)
    path = Path(receipts_dir) / f"paper_replication_{sealed['receipt_sha256'][:16]}.json"
    _atomic_write_text(path, json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    return path


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _check_block(value: Any, lo: float, hi: float, label: str, errors: list[str]) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        errors.append(f"{label}: non-numeric")
        return
    v = float(value)
    if math.isnan(v):
        return
    if not (lo - 1e-9 <= v <= hi + 1e-9):
        errors.append(f"{label}: {v} outside [{lo}, {hi}]")


def verify_replication_receipt(path: Path | str) -> dict[str, Any]:
    """Fail-closed standalone verification of a replication receipt file."""
    path = Path(path)
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "path": str(path), "errors": [f"unreadable: {exc}"]}
    if not isinstance(payload, dict):
        return {"valid": False, "path": str(path), "errors": ["not a JSON object"]}
    digest = payload.get("receipt_sha256")
    if not _is_sha256(digest):
        errors.append("missing/malformed receipt_sha256")
    else:
        content = {k: v for k, v in payload.items() if k != "receipt_sha256"}
        expected = hash_bytes(canonical_json_bytes(json.loads(canonical_json_bytes(content))))
        if digest != expected:
            errors.append("receipt_sha256 mismatch — payload was modified")
    for key in REQUIRED_TOP_KEYS:
        if key not in payload:
            errors.append(f"missing key: {key}")
    if payload.get("schema") != REPLICATION_SCHEMA:
        errors.append("schema mismatch")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim must be false")
    if payload.get("research_only") is not True:
        errors.append("research_only must be true")
    if payload.get("data_label") != "SYNTHETIC":
        errors.append("data_label must be SYNTHETIC")
    if not _is_sha256(payload.get("inputs_sha256")):
        errors.append("inputs_sha256 malformed")
    research_blob = {k: v for k, v in payload.items() if k != "live_pnl_claim"}
    if not family_blob_forbidden_metrics_absent(research_blob):
        errors.append("forbidden research metric keys present")
    strategies = payload.get("strategies")
    if not isinstance(strategies, dict) or not strategies:
        errors.append("strategies missing")
    else:
        for name, block in strategies.items():
            if not isinstance(block, dict):
                errors.append(f"{name}: not an object")
                continue
            scores = block.get("scores")
            if not isinstance(scores, dict) or not scores:
                errors.append(f"{name}: scores missing")
                continue
            for key, value in scores.items():
                if (
                    key.endswith("brier")
                    or key.endswith("brier_climatology")
                    or key.endswith("_ece")
                    or key == "ece"
                ):
                    _check_block(value, 0.0, 1.0, f"{name}.{key}", errors)
                elif key.endswith("mean_ic"):
                    _check_block(value, -1.0, 1.0, f"{name}.{key}", errors)
                elif key.endswith("p_value_nw") or key == "primary_p_value":
                    _check_block(value, 0.0, 1.0, f"{name}.{key}", errors)
            verdict = block.get("verdict")
            if not isinstance(verdict, dict):
                errors.append(f"{name}: verdict missing")
            else:
                v = verdict.get("verdict")
                if v not in {"replicated", "contradicted", "inconclusive", "untestable"}:
                    errors.append(f"{name}: bad verdict {v!r}")
                if v == "untestable" and not (
                    isinstance(verdict.get("untestable_reason"), str)
                    and verdict["untestable_reason"]
                ):
                    errors.append(f"{name}: untestable verdict lacks a reason")
                if v == "replicated" and not (
                    verdict.get("direction_consistent_with_paper") is True
                    and verdict.get("significant_at_5pct") is True
                ):
                    errors.append(f"{name}: verdict inconsistent with flags")
                _check_block(
                    verdict.get("primary_p_value"), 0.0, 1.0, f"{name}.primary_p_value", errors
                )
            descriptive = block.get("descriptive")
            if not isinstance(descriptive, dict):
                errors.append(f"{name}: descriptive block missing")
            elif any(not k.startswith("descriptive_") for k in descriptive):
                errors.append(f"{name}: unlabeled performance stats outside descriptive_")
    return {"valid": not errors, "path": str(path), "errors": errors}
