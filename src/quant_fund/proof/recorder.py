"""Data access recorder: the audit trail of every PIT vault read (DESIGN.md §5.1).

``InMemoryRecorder`` is attached to ``PitVault(recorder=...)``. The vault calls
``record`` **after** computing ``content_sha256`` on the returned frame, so the
proof commits to exactly the bytes the strategy consumed.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Protocol, runtime_checkable

from quant_fund.proofcore.contracts import (
    DataAccessRecord,
    DataManifestSummary,
    merkle_root_hex,
    sha256_hex_json,
)

__all__ = ["DataAccessRecorder", "InMemoryRecorder", "make_read_record"]


@runtime_checkable
class DataAccessRecorder(Protocol):
    """Anything a PitVault can report a read into (DESIGN.md §5.1)."""

    def record(self, read: DataAccessRecord) -> None: ...


def make_read_record(
    dataset: str,
    asof: datetime,
    *,
    params: dict[str, object] | None = None,
    rows: int,
    content_sha256: str,
) -> DataAccessRecord:
    """Build a DataAccessRecord, str-coercing params at the recorder boundary.

    ``DataAccessRecord.params`` values are strings only so canonical JSON is
    byte-stable (DESIGN.md §3 notes); coercion lives here, not in the vault.
    """
    coerced = {str(key): str(value) for key, value in (params or {}).items()}
    return DataAccessRecord(
        dataset=dataset,
        asof_utc=asof.isoformat(),
        params=coerced,
        rows=rows,
        content_sha256=content_sha256,
    )


def read_leaf_hash(read: DataAccessRecord) -> str:
    """Merkle leaf hash for one recorded read (shared by minter + verifier)."""
    return sha256_hex_json(read.model_dump(mode="json"))


class InMemoryRecorder:
    """Thread-safe in-memory read log passed into ``PitVault(recorder=...)``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reads: list[DataAccessRecord] = []

    @property
    def reads(self) -> list[DataAccessRecord]:
        """Snapshot of the recorded reads in record order."""
        with self._lock:
            return list(self._reads)

    def record(self, read: DataAccessRecord) -> None:
        with self._lock:
            self._reads.append(read)

    def manifest_summary(self) -> DataManifestSummary:
        """Merkle summary over the recorded reads (DESIGN.md §5.1).

        merkle_root_hex([sha256_hex_json(r.model_dump(mode='json')) for r in reads])
        """
        reads = self.reads
        return DataManifestSummary(
            reads=reads,
            merkle_root=merkle_root_hex([read_leaf_hash(read) for read in reads]),
            n_reads=len(reads),
        )
