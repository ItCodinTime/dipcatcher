"""Unit tests for proof.recorder (DESIGN.md §5.1)."""

from __future__ import annotations

import threading
from datetime import UTC, datetime

from quant_fund.proof.recorder import (
    DataAccessRecorder,
    InMemoryRecorder,
    make_read_record,
    read_leaf_hash,
)
from quant_fund.proofcore.contracts import (
    DataAccessRecord,
    merkle_root_hex,
    sha256_hex_bytes,
    sha256_hex_json,
)

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _record(tag: str = "silver/bars", rows: int = 3) -> DataAccessRecord:
    return make_read_record(
        tag,
        T0,
        params={"columns": ["close"], "limit": 10},
        rows=rows,
        content_sha256=sha256_hex_bytes(tag.encode()),
    )


def test_make_read_record_str_coerces_params() -> None:
    record = _record()
    assert record.params == {"columns": "['close']", "limit": "10"}
    assert all(isinstance(v, str) for v in record.params.values())
    assert record.asof_utc == T0.isoformat()


def test_recorder_satisfies_protocol() -> None:
    assert isinstance(InMemoryRecorder(), DataAccessRecorder)


def test_manifest_summary_matches_spec_formula() -> None:
    recorder = InMemoryRecorder()
    reads = [_record("a", 1), _record("b", 2), _record("c", 3)]
    for read in reads:
        recorder.record(read)
    summary = recorder.manifest_summary()
    expected = merkle_root_hex([sha256_hex_json(r.model_dump(mode="json")) for r in reads])
    assert summary.merkle_root == expected
    assert summary.n_reads == 3
    assert list(summary.reads) == reads
    assert read_leaf_hash(reads[0]) == sha256_hex_json(reads[0].model_dump(mode="json"))


def test_manifest_summary_empty_is_empty_hash() -> None:
    summary = InMemoryRecorder().manifest_summary()
    assert summary.n_reads == 0
    assert summary.merkle_root == sha256_hex_bytes(b"")


def test_recorder_thread_safe() -> None:
    recorder = InMemoryRecorder()

    def worker(n: int) -> None:
        for i in range(50):
            recorder.record(_record(f"ds{n}", i))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert recorder.manifest_summary().n_reads == 8 * 50


def test_reads_snapshot_isolation() -> None:
    recorder = InMemoryRecorder()
    recorder.record(_record())
    snapshot = recorder.reads
    snapshot.clear()
    assert recorder.manifest_summary().n_reads == 1
