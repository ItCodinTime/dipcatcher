"""Load and minimally verify dipcatcher receipts for corpus use.

A receipt is only eligible as fx-1 training evidence when it is explicitly
research-scoped: ``research_only`` true and ``live_pnl_claim`` false. Receipts
failing that check are excluded fail-closed — the model must never train on
live-performance implications.
"""

from __future__ import annotations

import hashlib
import json
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
    payload: dict = Field(default_factory=dict)

    @property
    def eligible(self) -> bool:
        """True iff the receipt is research-scoped and makes no live claim."""
        return self.research_only and not self.live_pnl_claim


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_receipts(receipts_dir: str | Path) -> list[ReceiptRecord]:
    """Load every ``*.json`` receipt under *receipts_dir*.

    Unparseable files are skipped (they cannot be verified, so they cannot
    be training evidence). Eligibility filtering happens at corpus build.
    """
    root = Path(receipts_dir)
    records: list[ReceiptRecord] = []
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        records.append(
            ReceiptRecord(
                path=str(path),
                sha256=_sha256(path),
                schema_name=str(payload.get("schema", "unknown")),
                research_only=bool(payload.get("research_only", False)),
                live_pnl_claim=bool(payload.get("live_pnl_claim", True)),
                disclaimer=str(payload.get("disclaimer", "")),
                payload=payload,
            )
        )
    return records
