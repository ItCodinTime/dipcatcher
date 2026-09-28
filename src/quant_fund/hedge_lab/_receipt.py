"""Shared seal + verify for hedge_lab lane receipts.

Every lane writer passes its payload through ``seal_receipt`` before writing;
``receipt_sha256`` covers the canonical JSON bytes excluding path keys, which
writers attach post-seal. ``verify_lane_receipt`` re-checks the honesty
contract and the digest on a written file — fail-closed on any drift.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

# Path keys are attached after sealing so filesystem layout never poisons the
# digest; both writer and verifier exclude them from the preimage.
PATH_KEYS = ("receipt_sha256", "receipt_path", "artifact_path", "metadata_path")

# Lane receipts are lab artifacts, not research-family scorecard blobs: they
# legitimately embed economic diagnostics (sharpe/sortino inside scoreboards),
# so the honesty fields are the guard, not forbidden-key scanning.
PAPER_CLAIMS = ("paper_backtest", "research_only")


def seal_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Copy of *receipt* with ``receipt_sha256`` bound over canonical JSON."""
    unsigned = {key: value for key, value in receipt.items() if key not in PATH_KEYS}
    return {**receipt, "receipt_sha256": hash_bytes(canonical_json_bytes(unsigned))}


def lane_receipt_seal_errors(receipt: Mapping[str, Any]) -> list[str]:
    """Digest errors for a written lane receipt."""
    unsigned = {key: value for key, value in receipt.items() if key not in PATH_KEYS}
    expected = hash_bytes(canonical_json_bytes(unsigned))
    return [] if receipt.get("receipt_sha256") == expected else ["receipt_sha256"]


def lane_receipt_contract_errors(receipt: Mapping[str, Any]) -> list[str]:
    """Honesty-contract errors every sealed hedge_lab lane receipt must satisfy."""
    errors: list[str] = []
    if receipt.get("research_only") is not True:
        errors.append("research_only")
    if receipt.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim")
    if receipt.get("execution_claim") not in PAPER_CLAIMS:
        errors.append("execution_claim")
    return errors


def verify_lane_receipt(path: str | Path) -> list[str]:
    """Fail-closed verification of a written hedge_lab lane receipt file."""
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"unreadable:{exc}"]
    if not isinstance(loaded, dict):
        return ["not_an_object"]
    return lane_receipt_contract_errors(loaded) + lane_receipt_seal_errors(loaded)
