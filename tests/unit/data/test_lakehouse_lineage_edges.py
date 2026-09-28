"""Lineage edge paths: drift taxonomy, render pinning/cycles, malformed
records and indexes, and code-identity guards."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from quant_fund.data.lakehouse.lineage import (
    INDEX_SCHEMA,
    LINEAGE_SCHEMA,
    LineageInput,
    format_dag,
    record_lineage,
    verify_lineage,
)
from quant_fund.data.lakehouse.store import write_partitioned_bars
from quant_fund.schemas.errors import DataContractError
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

pytestmark = pytest.mark.synthetic

T0 = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
T1 = datetime(2024, 1, 2, 14, 31, tzinfo=UTC)


def _bars() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "security_id": ["A", "A"],
            "symbol": ["AAPL", "AAPL"],
            "event_time": [T0, T1],
            "available_time": [T1, T1],
            "source": ["stooq", "stooq"],
            "open": [10.0, 11.0],
            "high": [12.0, 12.0],
            "low": [9.0, 10.0],
            "close": [11.0, 11.5],
            "volume": [100.0, 200.0],
        }
    )


def _lake(tmp_path: Path, *names: str) -> tuple[Path, dict[str, str]]:
    lake = tmp_path / "lake"
    snaps = {
        name: write_partitioned_bars(lake, _bars(), dataset=name).snapshot_id for name in names
    }
    return lake, snaps


def _record_path(lake: Path, record_id: str) -> Path:
    return lake / "lineage" / "records" / f"{record_id}.json"


def _index_path(lake: Path) -> Path:
    return lake / "lineage" / "index.json"


def _write_index(lake: Path, mapping: dict[str, str]) -> None:
    body = canonical_json_bytes({"schema": INDEX_SCHEMA, "datasets": mapping})
    _index_path(lake).write_bytes(body)


class TestFormatDag:
    def test_no_lineage_raises(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        with pytest.raises(DataContractError, match="no lineage"):
            format_dag(lake, "anything")

    def test_input_without_record_renders_root_snapshot(self, tmp_path: Path) -> None:
        lake, snaps = _lake(tmp_path, "derived", "raw")
        record_lineage(
            lake,
            dataset="derived",
            snapshot_id=snaps["derived"],
            inputs=(LineageInput("raw", snaps["raw"]),),
            code="import",
        )
        text = format_dag(lake, "derived")
        assert "raw" in text and "(root snapshot)" in text and snaps["raw"] in text

    def test_pinned_snapshot_mismatch_renders_root(self, tmp_path: Path) -> None:
        lake, snaps = _lake(tmp_path, "derived", "raw")
        first = record_lineage(lake, dataset="raw", snapshot_id=snaps["raw"], code="v1")
        snap2 = write_partitioned_bars(
            lake, _bars().with_columns(pl.col("volume") * 2.0), dataset="raw"
        ).snapshot_id
        record_lineage(lake, dataset="raw", snapshot_id=snap2, code="v2")
        record_lineage(
            lake,
            dataset="derived",
            snapshot_id=snaps["derived"],
            inputs=(LineageInput("raw", first.snapshot_id),),
            code="import",
        )
        text = format_dag(lake, "derived")
        assert "(root snapshot)" in text and first.snapshot_id in text

    def test_input_cycle_renders_cycle_marker(self, tmp_path: Path) -> None:
        lake, snaps = _lake(tmp_path, "a", "b")
        record_lineage(
            lake,
            dataset="a",
            snapshot_id=snaps["a"],
            inputs=(LineageInput("b", snaps["b"]),),
            code="x",
        )
        record_lineage(
            lake,
            dataset="b",
            snapshot_id=snaps["b"],
            inputs=(LineageInput("a", snaps["a"]),),
            code="x",
        )
        assert "(cycle)" in format_dag(lake, "a")


class TestVerifyDrift:
    def test_missing_dataset_reports_missing(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        report = verify_lineage(lake, "nope")
        assert report.ok is False
        assert report.errors == ("lineage_missing:nope", "lineage_missing:nope")

    def test_unreadable_record(self, tmp_path: Path) -> None:
        lake, snaps = _lake(tmp_path, "bars")
        rec = record_lineage(lake, dataset="bars", snapshot_id=snaps["bars"], code="x")
        _record_path(lake, rec.record_id).write_text('{"schema": "bogus"}')
        report = verify_lineage(lake, "bars")
        assert any(e.startswith("lineage_record_unreadable:bars") for e in report.errors)

    def test_index_points_at_other_dataset(self, tmp_path: Path) -> None:
        lake, snaps = _lake(tmp_path, "a", "b")
        record_lineage(lake, dataset="a", snapshot_id=snaps["a"], code="x")
        rb = record_lineage(lake, dataset="b", snapshot_id=snaps["b"], code="x")
        _write_index(lake, {"a": rb.record_id, "b": rb.record_id})
        report = verify_lineage(lake, "a")
        assert any(e.startswith("lineage_index_mismatch:a") for e in report.errors)

    def test_cycle_reports_cycle(self, tmp_path: Path) -> None:
        lake, snaps = _lake(tmp_path, "a", "b")
        record_lineage(
            lake,
            dataset="a",
            snapshot_id=snaps["a"],
            inputs=(LineageInput("b", snaps["b"]),),
            code="x",
        )
        record_lineage(
            lake,
            dataset="b",
            snapshot_id=snaps["b"],
            inputs=(LineageInput("a", snaps["a"]),),
            code="x",
        )
        report = verify_lineage(lake, "a")
        assert any(e.startswith("lineage_cycle:") for e in report.errors)

    def test_snapshot_unreadable(self, tmp_path: Path) -> None:
        lake, snaps = _lake(tmp_path, "bars")
        record_lineage(lake, dataset="bars", snapshot_id=snaps["bars"], code="x")
        (lake / "snapshots" / f"{snaps['bars']}.json").write_text("garbage")
        report = verify_lineage(lake, "bars")
        assert any(e.startswith("snapshot_unreadable:") for e in report.errors)

    def test_object_missing(self, tmp_path: Path) -> None:
        from quant_fund.data.lakehouse.store import load_snapshot

        lake, snaps = _lake(tmp_path, "bars")
        record_lineage(lake, dataset="bars", snapshot_id=snaps["bars"], code="x")
        snapshot = load_snapshot(lake, snaps["bars"])
        (lake / snapshot.files[0].object_path).unlink()
        report = verify_lineage(lake, "bars")
        assert any(e.startswith("object_missing:") for e in report.errors)

    def test_code_missing(self, tmp_path: Path) -> None:
        lake, snaps = _lake(tmp_path, "bars")
        code = tmp_path / "producer.py"
        code.write_text("x = 1\n")
        record_lineage(lake, dataset="bars", snapshot_id=snaps["bars"], code=code)
        code.unlink()
        report = verify_lineage(lake, "bars")
        assert any(e.startswith("code_missing:") for e in report.errors)

    def test_input_snapshot_missing(self, tmp_path: Path) -> None:
        lake, snaps = _lake(tmp_path, "derived", "raw")
        record_lineage(
            lake,
            dataset="derived",
            snapshot_id=snaps["derived"],
            inputs=(LineageInput("raw", snaps["raw"]),),
            code="x",
        )
        (lake / "snapshots" / f"{snaps['raw']}.json").unlink()
        report = verify_lineage(lake, "derived")
        assert any(e.startswith("input_snapshot_missing:raw") for e in report.errors)

    def test_child_record_unreadable(self, tmp_path: Path) -> None:
        lake, snaps = _lake(tmp_path, "derived", "raw")
        rb = record_lineage(lake, dataset="raw", snapshot_id=snaps["raw"], code="x")
        record_lineage(
            lake,
            dataset="derived",
            snapshot_id=snaps["derived"],
            inputs=(LineageInput("raw", snaps["raw"]),),
            code="x",
        )
        _record_path(lake, rb.record_id).write_text('{"schema": "bogus"}')
        report = verify_lineage(lake, "derived")
        assert any(e.startswith("lineage_record_unreadable:raw") for e in report.errors)

    def test_child_without_index_skips(self, tmp_path: Path) -> None:
        lake, snaps = _lake(tmp_path, "derived", "raw")
        record_lineage(
            lake,
            dataset="derived",
            snapshot_id=snaps["derived"],
            inputs=(LineageInput("raw", snaps["raw"]),),
            code="x",
        )
        # raw has a snapshot but no lineage record at all.
        assert verify_lineage(lake, "derived").ok is True


class TestMalformedDocuments:
    def _craft_record(self, lake: Path, **overrides: object) -> str:
        body: dict[str, object] = {
            "schema": LINEAGE_SCHEMA,
            "dataset": "crafted",
            "snapshot_id": "s" * 4,
            "inputs": [],
            "code_hash": "h" * 4,
            "code_path": None,
            "parameters": {},
        }
        body.update(overrides)
        record_id = hash_bytes(canonical_json_bytes(body))
        payload = dict(body)
        payload["record_id"] = record_id
        path = _record_path(lake, record_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json.dumps(payload, sort_keys=True).encode())
        return record_id

    def test_record_id_mismatch_with_bytes(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        record_id = self._craft_record(lake)
        path = _record_path(lake, record_id)
        payload = json.loads(path.read_text())
        payload["record_id"] = "0" * 64
        path.write_bytes(json.dumps(payload, sort_keys=True).encode())
        _write_index(lake, {"crafted": record_id})
        report = verify_lineage(lake, "crafted")
        assert any("does not match its bytes" in e for e in report.errors)

    def test_inputs_malformed(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        record_id = self._craft_record(lake, inputs="not-a-list")
        _write_index(lake, {"crafted": record_id})
        report = verify_lineage(lake, "crafted")
        assert any("inputs are malformed" in e for e in report.errors)

    def test_parameters_malformed(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        record_id = self._craft_record(lake, parameters=5)
        _write_index(lake, {"crafted": record_id})
        report = verify_lineage(lake, "crafted")
        assert any("parameters are malformed" in e for e in report.errors)

    def test_code_path_malformed(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        record_id = self._craft_record(lake, code_path=7)
        _write_index(lake, {"crafted": record_id})
        report = verify_lineage(lake, "crafted")
        assert any("code_path is malformed" in e for e in report.errors)

    def test_index_malformed_schema(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        _index_path(lake).parent.mkdir(parents=True, exist_ok=True)
        _index_path(lake).write_text(json.dumps({"schema": "bogus"}))
        with pytest.raises(DataContractError, match="index is malformed"):
            verify_lineage(lake)

    def test_index_datasets_not_dict(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake"
        _index_path(lake).parent.mkdir(parents=True, exist_ok=True)
        _index_path(lake).write_text(json.dumps({"schema": INDEX_SCHEMA, "datasets": [1, 2]}))
        with pytest.raises(DataContractError, match="index is malformed"):
            verify_lineage(lake)


class TestRecordAndCodeIdentity:
    def test_record_conflict_bytes(self, tmp_path: Path) -> None:
        lake, snaps = _lake(tmp_path, "bars")
        rec = record_lineage(lake, dataset="bars", snapshot_id=snaps["bars"], code="x")
        _record_path(lake, rec.record_id).write_bytes(b"different")
        with pytest.raises(DataContractError, match="different bytes"):
            record_lineage(lake, dataset="bars", snapshot_id=snaps["bars"], code="x")

    def test_missing_input_snapshot_rejected(self, tmp_path: Path) -> None:
        lake, snaps = _lake(tmp_path, "bars")
        with pytest.raises(DataContractError):
            record_lineage(
                lake,
                dataset="bars",
                snapshot_id=snaps["bars"],
                inputs=(LineageInput("ghost", "0" * 64),),
                code="x",
            )

    def test_lfs_pointer_code_rejected(self, tmp_path: Path) -> None:
        lake, snaps = _lake(tmp_path, "bars")
        pointer = tmp_path / "model.bin"
        pointer.write_text(
            "version https://git-lfs.github.com/spec/v1\noid sha256:" + "ab" * 32 + "\nsize 123\n"
        )
        with pytest.raises(DataContractError, match="file bytes"):
            record_lineage(lake, dataset="bars", snapshot_id=snaps["bars"], code=pointer)

    def test_unreadable_callable_source_rejected(self, tmp_path: Path) -> None:
        lake, snaps = _lake(tmp_path, "bars")
        with pytest.raises(DataContractError, match="callable source"):
            record_lineage(lake, dataset="bars", snapshot_id=snaps["bars"], code=len)

    def test_bad_code_type_rejected(self, tmp_path: Path) -> None:
        lake, snaps = _lake(tmp_path, "bars")
        with pytest.raises(DataContractError, match="path, callable, or source"):
            record_lineage(lake, dataset="bars", snapshot_id=snaps["bars"], code=5)  # type: ignore[arg-type]
