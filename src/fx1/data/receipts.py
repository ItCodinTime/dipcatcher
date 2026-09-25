"""Load and minimally verify dipcatcher receipts for corpus use.

A receipt is only eligible as fx-1 training evidence when it is explicitly
research-scoped: ``research_only`` true and ``live_pnl_claim`` false. Receipts
failing that check are excluded fail-closed — the model must never train on
live-performance implications.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, Field


class ReceiptRecord(BaseModel):
    """Minimal verified view of a dipcatcher receipt file."""

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
    payload: dict = Field(default_factory=dict)

    @property
    def eligible(self) -> bool:
        """True iff the receipt is research-scoped and makes no live claim."""
        return self.research_only and not self.live_pnl_claim


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _eligibility(payload: dict) -> tuple[bool, bool]:
    """Resolve (research_only, live_pnl_claim) across receipt schemas.

    Two schemas are recognized, both explicit:

    - boolean contract: ``research_only: true`` + ``live_pnl_claim: false``.
    - lab run-manifest contract: ``claim: "research_only"`` declares the
      research scope; an explicit ``live_pnl_claim`` key always wins, and in
      its absence the research-only declaration implies no live claim.

    Anything else fails closed: not research-scoped, live claim assumed.
    """
    claim = payload.get("claim")
    research_only = bool(payload.get("research_only", False)) or (claim == "research_only")
    if "live_pnl_claim" in payload:
        live_pnl_claim = bool(payload["live_pnl_claim"])
    else:
        live_pnl_claim = claim != "research_only"
    return research_only, live_pnl_claim


def load_receipts(
    receipts_dir: str | Path | Iterable[str | Path],
) -> list[ReceiptRecord]:
    """Load every ``*.json`` receipt under one or more receipt directories.

    Unparseable files are skipped (they cannot be verified, so they cannot
    be training evidence). Eligibility filtering happens at corpus build.
    """
    if isinstance(receipts_dir, str | Path):
        roots = [Path(receipts_dir)]
    else:
        roots = [Path(d) for d in receipts_dir]
    records: list[ReceiptRecord] = []
    for root in roots:
        for path in sorted(root.rglob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            research_only, live_pnl_claim = _eligibility(payload)
            records.append(
                ReceiptRecord(
                    path=str(path),
                    sha256=_sha256(path),
                    schema_name=str(
                        payload.get("schema", payload.get("schema_version", "unknown"))
                    ),
                    research_only=research_only,
                    live_pnl_claim=live_pnl_claim,
                    disclaimer=str(payload.get("disclaimer", "")),
                    evidence_class=("synthetic" if payload.get("synthetic") else "research"),
                    payload=payload,
                )
            )
    return records
