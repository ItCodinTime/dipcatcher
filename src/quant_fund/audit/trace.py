"""Bind a research receipt to the audit ledger without modifying the receipt.

The digest is ``quant_fund.research.verify._receipt_digest``, the same hash
``verify-research`` uses. A published number is located by a dotted path inside
that receipt. The link is the ledger entry whose ``receipt_sha256`` matches,
plus an inclusion proof under the signed Merkle root. Code identity is the
git revision and worktree hash already stored on the receipt; this module
does not re-seal them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quant_fund.audit.errors import AuditError
from quant_fund.audit.ledger import AuditLedger
from quant_fund.audit.merkle import hash_leaf, verify_inclusion
from quant_fund.audit.verify import verify_ledger
from quant_fund.research.verify import _receipt_digest
from quant_fund.utils.reproducibility import git_revision, git_worktree_sha256

LINK_FIELDS = (
    "run_id",
    "git_revision",
    "git_worktree_sha256",
    "config_sha256",
    "dataset_sha256",
    "dataset_content_sha256",
    "northset_inputs_sha256",
    "code_sha256",
)


def receipt_digest(notebook: dict[str, Any]) -> str:
    """Hash a receipt the same way the research verifier hashes it."""
    return _receipt_digest(notebook)


def lookup_path(document: Any, metric_path: str) -> Any:
    """Walk a dotted path. Numeric parts index lists. Missing steps raise KeyError."""
    if not metric_path or metric_path.startswith(".") or metric_path.endswith("."):
        raise KeyError(metric_path)
    current = document
    for part in metric_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if index >= len(current):
                raise KeyError(metric_path)
            current = current[index]
            continue
        raise KeyError(metric_path)
    return current


def trace_published_number(
    receipt_path: Path | str,
    metric_path: str,
    ledger: AuditLedger,
    *,
    trust_public_key: bytes | None = None,
    verify_receipt: bool = False,
    compare_worktree: bool = False,
) -> dict[str, Any]:
    """Trace one number inside a receipt to its ledger entry and code hashes.

    ``linked`` is true only when the ledger verifies, the receipt digest matches
    an entry, the provenance fields match, the number exists, and the inclusion
    proof rebuilds the signed root. ``verify_receipt`` embeds the existing
    ``verify-research`` result and does not change ``linked``.
    """
    path = Path(receipt_path)
    before = path.read_bytes()
    try:
        parsed = json.loads(before)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return _unlinked(metric_path, [f"receipt_unreadable:{exc}"])
    if not isinstance(parsed, dict):
        return _unlinked(metric_path, ["receipt_not_object"])
    notebook: dict[str, Any] = parsed
    try:
        digest = receipt_digest(notebook)
    except (TypeError, ValueError) as exc:
        return _unlinked(metric_path, [f"receipt_digest_failed:{exc}"])

    errors: list[str] = []
    try:
        value = lookup_path(notebook, metric_path)
    except KeyError:
        value = None
        errors.append("metric_not_found")

    report = verify_ledger(ledger.root, trust_public_key=trust_public_key)
    if not report["valid"]:
        errors.append("ledger_invalid")
        errors.extend(str(item) for item in report["errors"])

    provenance = notebook.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
        errors.append("provenance_missing")

    try:
        entries = ledger.entries()
    except AuditError as exc:
        entries = []
        errors.append("ledger_unreadable")
        errors.extend(exc.errors)

    matches = [
        entry
        for entry in entries
        if entry.kind == "research_run" and entry.payload.get("receipt_sha256") == digest
    ]
    if not matches:
        errors.append("receipt_not_in_ledger")
        entry = None
    else:
        entry = matches[-1]
        for field in LINK_FIELDS:
            recorded = entry.payload.get(field)
            current = provenance.get(field)
            if current is None and field == "code_sha256":
                current = notebook.get("code_sha256")
            if recorded != current:
                errors.append(f"provenance_mismatch:{field}")

    inclusion: dict[str, Any] | None = None
    if entry is not None and report.get("merkle_root") and report.get("tree_size"):
        try:
            proof = ledger.inclusion_proof(entry.index)
            root = bytes.fromhex(str(report["merkle_root"]))
            inclusion_ok = verify_inclusion(
                hash_leaf(entry.preimage()),
                entry.index,
                proof,
                int(report["tree_size"]),
                root,
            )
        except (AuditError, ValueError):
            inclusion_ok = False
            proof = []
        if not inclusion_ok:
            errors.append("inclusion_failed")
        inclusion = {
            "index": entry.index,
            "tree_size": report["tree_size"],
            "merkle_root": report["merkle_root"],
            "proof": [node.hex() for node in proof],
            "valid": inclusion_ok,
        }

    checkout_revision = git_revision()
    recorded_revision = provenance.get("git_revision")
    result: dict[str, Any] = {
        "linked": not errors,
        "metric_path": metric_path,
        "value": value,
        "receipt_sha256": digest,
        "ledger_index": None if entry is None else entry.index,
        "entry_hash": None if entry is None else entry.entry_hash,
        "git_revision": recorded_revision,
        "git_worktree_sha256": provenance.get("git_worktree_sha256"),
        "config_sha256": provenance.get("config_sha256"),
        "dataset_sha256": provenance.get("dataset_sha256"),
        "dataset_content_sha256": provenance.get("dataset_content_sha256"),
        "northset_inputs_sha256": provenance.get("northset_inputs_sha256"),
        "code_sha256": provenance.get("code_sha256", notebook.get("code_sha256")),
        "checkout_git_revision": checkout_revision,
        "checkout_matches_recorded_revision": checkout_revision == recorded_revision,
        "inclusion": inclusion,
        "ledger": report,
        "errors": errors,
        "live_pnl_claim": False,
    }
    if compare_worktree:
        checkout_tree = git_worktree_sha256()
        result["checkout_git_worktree_sha256"] = checkout_tree
        result["checkout_matches_recorded_worktree"] = checkout_tree == provenance.get(
            "git_worktree_sha256"
        )
    if verify_receipt:
        from quant_fund.research.verify import verify_research_artifact

        result["research_verifier"] = verify_research_artifact(path)
    if path.read_bytes() != before:
        result["linked"] = False
        result["errors"] = [*errors, "receipt_mutated"]
    return result


def _unlinked(metric_path: str, errors: list[str]) -> dict[str, Any]:
    return {
        "linked": False,
        "metric_path": metric_path,
        "value": None,
        "errors": errors,
        "live_pnl_claim": False,
    }
