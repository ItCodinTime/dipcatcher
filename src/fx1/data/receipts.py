"""Load and verify dipcatcher receipts for corpus use.

A receipt is only eligible as positive fx-1 training evidence when the public
``dipcatcher verify-receipt`` boundary validates its seal, it is explicitly
research-scoped, it makes no live claim, and it is not a failed/blocked run.
The subprocess boundary is intentional: corpus construction stays independent
of harness internals while still consuming the harness's schema-dispatched
verifier.  Every rejected or unreadable file remains auditable as a negative
example instead of disappearing silently.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from fx1.honesty import FORBIDDEN_HEADLINE_TOKENS


class ReceiptRecord(BaseModel):
    """Verified view of a dipcatcher receipt file."""

    path: str
    sha256: str = Field(description="SHA-256 of the raw receipt file bytes")
    schema_name: str = Field(default="unknown")
    research_only: bool
    live_pnl_claim: bool
    disclaimer: str = ""
    evidence_class: str = Field(
        default="research",
        description="Explicit evidence class: research | synthetic (from the "
        "payload's `synthetic` flag). Synthetic evidence is eligible only "
        "when research-scoped and is always labeled as simulated.",
    )
    verified: bool = False
    verification_errors: list[str] = Field(default_factory=list)
    verdict: str | None = None
    result_summary: dict[str, Any] = Field(default_factory=dict)
    payload: dict = Field(default_factory=dict)

    @property
    def eligible(self) -> bool:
        """True iff the receipt is verified, research-scoped, and passing."""
        return (
            self.verified
            and self.research_only
            and not self.live_pnl_claim
            and self.verdict not in {"fail", "blocked"}
        )


class ReceiptVerifier(Protocol):
    """Public verifier boundary used by the corpus loader."""

    def __call__(self, path: Path) -> dict[str, Any]: ...


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _eligibility(payload: dict, *, verified: bool) -> tuple[bool, bool]:
    """Resolve (research_only, live_pnl_claim) across receipt schemas.

    Two schemas are recognized, both explicit:

    - boolean contract: ``research_only: true`` + ``live_pnl_claim: false``.
    - lab run-manifest contract: ``claim: "research_only"`` declares the
      research scope; an explicit ``live_pnl_claim`` key always wins, and in
      its absence the research-only declaration implies no live claim.

    Verified ``receipt.v2`` envelopes and explicitly SYNTHETIC receipts are
    research evidence by construction. Anything else fails closed: not
    research-scoped, live claim assumed.
    """
    claim = payload.get("claim")
    research_only = (
        bool(payload.get("research_only", False))
        or claim == "research_only"
        or (verified and payload.get("schema") == "receipt.v2")
        or (verified and str(payload.get("data_label", "")).upper() == "SYNTHETIC")
    )
    if "live_pnl_claim" in payload:
        live_pnl_claim = bool(payload["live_pnl_claim"])
    else:
        live_pnl_claim = claim != "research_only"
    return research_only, live_pnl_claim


def _verify_with_harness(path: Path) -> dict[str, Any]:
    """Run the public receipt verifier without importing harness internals."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "quant_fund.cli.main", "verify-receipt", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"valid": False, "errors": [f"verifier_unavailable:{exc.__class__.__name__}"]}
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        detail = proc.stderr.strip().splitlines()
        suffix = detail[-1] if detail else f"exit_{proc.returncode}"
        return {"valid": False, "errors": [f"verifier_unavailable:{suffix}"]}
    if not isinstance(result, dict):
        return {"valid": False, "errors": ["verifier_returned_non_object"]}
    return result


_FORBIDDEN_SUMMARY_TOKENS = FORBIDDEN_HEADLINE_TOKENS
_SUMMARY_KEYS = frozenset(
    {
        "baseline_recovery",
        "brier",
        "coverage",
        "crps",
        "development_eligible",
        "ece",
        "economic_evidence_gate",
        "eligible",
        "kind",
        "kupiec",
        "likelihood",
        "metrics",
        "n_error_rows",
        "n_eval",
        "n_rows",
        "n_train",
        "pinball",
        "pit",
        "promote",
        "qlike",
        "selected",
        "selected_band",
        "status",
        "verdict",
    }
)


def _safe_summary_value(value: object, *, depth: int = 0) -> object:
    """Bound a receipt-derived summary and remove forbidden headline keys."""
    if depth > 3:
        return "<nested>"
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return {"count": len(value)}
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, child in value.items():
            name = str(key)
            tokens = {token for token in name.lower().replace("-", "_").split("_") if token}
            if tokens & _FORBIDDEN_SUMMARY_TOKENS or name == "live_pnl_claim":
                continue
            metric_markers = (
                "crps",
                "pinball",
                "pit",
                "qlike",
                "brier",
                "ece",
                "coverage",
                "kupiec",
            )
            if depth == 0 and not (
                name in _SUMMARY_KEYS or any(marker in name.lower() for marker in metric_markers)
            ):
                continue
            result[name] = _safe_summary_value(child, depth=depth + 1)
        return result
    return str(value)


def _result_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract a compact, honesty-safe result summary across receipt schemas."""
    body = payload.get("payload") if payload.get("schema") == "receipt.v2" else payload
    if not isinstance(body, dict):
        return {}
    correctness = body.get("correctness")
    if isinstance(correctness, dict) and correctness:
        safe = _safe_summary_value(correctness, depth=1)
        return dict(safe) if isinstance(safe, dict) else {"correctness": safe}
    safe = _safe_summary_value(body)
    summary = dict(safe) if isinstance(safe, dict) else {}
    results = body.get("results")
    if isinstance(results, list):
        summary["results_count"] = len(results)
    return summary


def _evidence_class(payload: dict[str, Any]) -> str:
    label = payload.get("data_label")
    if payload.get("synthetic") is True or str(label).upper() == "SYNTHETIC":
        return "synthetic"
    return "research"


def load_receipts(
    receipts_dir: str | Path | Iterable[str | Path],
    *,
    verifier: ReceiptVerifier | None = None,
) -> list[ReceiptRecord]:
    """Load every ``*.json`` receipt under one or more receipt directories.

    Unparseable and invalid files are retained as negative evidence with a
    verifier reason. Eligibility filtering happens at corpus build.
    """
    if isinstance(receipts_dir, str | Path):
        roots = [Path(receipts_dir)]
    else:
        roots = [Path(d) for d in receipts_dir]
    verify = verifier or _verify_with_harness
    records: list[ReceiptRecord] = []
    for root in roots:
        for path in sorted(root.rglob("*.json")):
            raw_sha256 = _sha256(path)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                records.append(
                    ReceiptRecord(
                        path=str(path),
                        sha256=raw_sha256,
                        schema_name="unreadable",
                        research_only=False,
                        live_pnl_claim=True,
                        verified=False,
                        verification_errors=[f"receipt_unreadable:{exc.__class__.__name__}"],
                    )
                )
                continue
            if not isinstance(payload, dict):
                records.append(
                    ReceiptRecord(
                        path=str(path),
                        sha256=raw_sha256,
                        schema_name="unknown",
                        research_only=False,
                        live_pnl_claim=True,
                        verified=False,
                        verification_errors=["receipt_not_object"],
                    )
                )
                continue
            try:
                verification = verify(path)
            except Exception as exc:  # verifier adapters are an external boundary
                verification = {
                    "valid": False,
                    "errors": [f"verifier_failed:{exc.__class__.__name__}"],
                }
            verified = verification.get("valid") is True
            raw_errors = verification.get("errors", [])
            errors = (
                [str(item) for item in raw_errors]
                if isinstance(raw_errors, list)
                else ["verifier_errors_invalid"]
            )
            research_only, live_pnl_claim = _eligibility(payload, verified=verified)
            verdict_value = verification.get("verdict", payload.get("verdict"))
            verdict = str(verdict_value).lower() if verdict_value is not None else None
            records.append(
                ReceiptRecord(
                    path=str(path),
                    sha256=raw_sha256,
                    schema_name=str(
                        payload.get("schema", payload.get("schema_version", "unknown"))
                    ),
                    research_only=research_only,
                    live_pnl_claim=live_pnl_claim,
                    disclaimer=str(payload.get("disclaimer", "")),
                    evidence_class=_evidence_class(payload),
                    verified=verified,
                    verification_errors=errors,
                    verdict=verdict,
                    result_summary=_result_summary(payload),
                    payload=payload,
                )
            )
    return records
