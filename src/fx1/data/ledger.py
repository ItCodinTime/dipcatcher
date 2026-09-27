"""The public corpus ledger — out-opening OLMo.

Even OLMo stops at publishing artifacts. fx-1 publishes a **hash-chained,
append-only ledger** recording, per training example:
source artifact hash → transformation code hash → example hash, plus every
quality-gate exclusion with the rule that caused it. Anyone can audit the
training set's construction without downloading the training set, and can
challenge any example's lineage.

The chain is tamper-evident: each entry's hash commits to the previous
entry's hash, so rewriting history breaks every subsequent link.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

GENESIS = "0" * 64


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class LedgerEntry(BaseModel):
    """One link in the corpus ledger chain."""

    index: int
    utc: str
    kind: str = Field(description="example | exclusion | gate_decision")
    source_sha256: str = Field(min_length=64, max_length=64)
    transform_sha256: str = Field(description="SHA-256 of the code/config that produced this entry")
    example_sha256: str | None = None
    rule: str | None = Field(default=None, description="quality-gate rule name for exclusions")
    prev_hash: str = Field(min_length=64, max_length=64)
    entry_hash: str = Field(min_length=64, max_length=64)

    @staticmethod
    def compute_hash(
        *,
        index: int,
        utc: str,
        kind: str,
        source_sha256: str,
        transform_sha256: str,
        example_sha256: str | None,
        rule: str | None,
        prev_hash: str,
    ) -> str:
        payload = json.dumps(
            {
                "index": index,
                "utc": utc,
                "kind": kind,
                "source_sha256": source_sha256,
                "transform_sha256": transform_sha256,
                "example_sha256": example_sha256,
                "rule": rule,
                "prev_hash": prev_hash,
            },
            sort_keys=True,
        )
        return _sha(payload)


class CorpusLedger:
    """Append-only, hash-chained corpus construction ledger."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._entries: list[LedgerEntry] = []
        if self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._entries.append(LedgerEntry.model_validate_json(line))

    def _append(
        self,
        *,
        kind: str,
        source_sha256: str,
        transform_sha256: str,
        example_sha256: str | None = None,
        rule: str | None = None,
    ) -> LedgerEntry:
        prev = self._entries[-1].entry_hash if self._entries else GENESIS
        utc = datetime.now(UTC).isoformat()
        entry_hash = LedgerEntry.compute_hash(
            index=len(self._entries),
            utc=utc,
            kind=kind,
            source_sha256=source_sha256,
            transform_sha256=transform_sha256,
            example_sha256=example_sha256,
            rule=rule,
            prev_hash=prev,
        )
        entry = LedgerEntry(
            index=len(self._entries),
            utc=utc,
            kind=kind,
            source_sha256=source_sha256,
            transform_sha256=transform_sha256,
            example_sha256=example_sha256,
            rule=rule,
            prev_hash=prev,
            entry_hash=entry_hash,
        )
        self._entries.append(entry)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")
        return entry

    def record_example(
        self, *, source_sha256: str, transform_sha256: str, example_sha256: str
    ) -> LedgerEntry:
        return self._append(
            kind="example",
            source_sha256=source_sha256,
            transform_sha256=transform_sha256,
            example_sha256=example_sha256,
        )

    def record_exclusion(
        self, *, source_sha256: str, transform_sha256: str, rule: str
    ) -> LedgerEntry:
        return self._append(
            kind="exclusion",
            source_sha256=source_sha256,
            transform_sha256=transform_sha256,
            rule=rule,
        )

    def verify_chain(self) -> bool:
        """Fail-closed: every link must hash correctly and chain to its parent."""
        prev = GENESIS
        for expected_index, entry in enumerate(self._entries):
            if entry.index != expected_index:
                return False
            if entry.prev_hash != prev:
                return False
            recomputed = LedgerEntry.compute_hash(
                index=entry.index,
                utc=entry.utc,
                kind=entry.kind,
                source_sha256=entry.source_sha256,
                transform_sha256=entry.transform_sha256,
                example_sha256=entry.example_sha256,
                rule=entry.rule,
                prev_hash=entry.prev_hash,
            )
            if recomputed != entry.entry_hash:
                return False
            prev = entry.entry_hash
        return True

    def audit_export(self) -> dict:
        """Public audit view: counts, rules fired, chain head — no raw data."""
        rules: dict[str, int] = {}
        for entry in self._entries:
            if entry.kind == "exclusion" and entry.rule:
                rules[entry.rule] = rules.get(entry.rule, 0) + 1
        return {
            "entries": len(self._entries),
            "examples": sum(1 for e in self._entries if e.kind == "example"),
            "exclusions": sum(1 for e in self._entries if e.kind == "exclusion"),
            "exclusion_rules": rules,
            "chain_head": self._entries[-1].entry_hash if self._entries else GENESIS,
            "chain_valid": self.verify_chain(),
        }
