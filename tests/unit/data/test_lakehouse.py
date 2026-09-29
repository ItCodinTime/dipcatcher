"""Content-addressed lake: snapshots, lineage, as-of queries, quality, migration."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.config.models import UniverseConfig
from quant_fund.data.lakehouse.lineage import (
    LineageInput,
    format_dag,
    record_lineage,
    verify_lineage,
)
from quant_fund.data.lakehouse.migrate import import_files
from quant_fund.data.lakehouse.panels import lagged_return_panel_bytes
from quant_fund.data.lakehouse.quality import (
    QualityThresholdError,
    QualityThresholds,
    quality_report,
    quality_report_path,
)
from quant_fund.data.lakehouse.query import asof_bars, asof_observations, universe_asof
from quant_fund.data.lakehouse.receipts import snapshot_receipt_fields
from quant_fund.data.lakehouse.store import (
    commit_snapshot,
    file_entry_for_object,
    load_snapshot,
    put_file,
    write_partitioned_bars,
)
from quant_fund.research.catalog.constants import FORBIDDEN_RESEARCH_METRIC_KEYS
from quant_fund.schemas.errors import DataContractError, LeakageError, PointInTimeError
from quant_fund.utils.hashing import hash_bytes
from quant_fund.utils.reproducibility import content_address

T0 = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
T1 = datetime(2024, 1, 2, 14, 31, tzinfo=UTC)
DAY = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
NEXT = datetime(2024, 1, 3, 21, 0, tzinfo=UTC)


def _bars() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "security_id": ["A", "A", "B"],
            "symbol": ["AAPL", "AAPL", "MSFT"],
            "event_time": [T0, T1, T0],
            "available_time": [T1, T1, T1],
            "source": ["stooq", "stooq", "stooq"],
            "open": [10.0, 11.0, 20.0],
            "high": [12.0, 12.0, 21.0],
            "low": [9.0, 10.0, 19.0],
            "close": [11.0, 11.5, 20.5],
            "volume": [100.0, 110.0, 80.0],
        }
    )


def test_content_address_hashes_bytes_and_lfs_oid(tmp_path: Path) -> None:
    blob = tmp_path / "bars.parquet"
    blob.write_bytes(b"parquet-bytes")
    address = content_address(blob)
    assert address.kind == "blob"
    assert address.content_sha256 == address.stored_sha256 == hash_bytes(b"parquet-bytes")

    oid = "ab" * 32
    pointer = tmp_path / "features.parquet"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n" + f"oid sha256:{oid}\n" + "size 42\n",
        encoding="ascii",
    )
    lfs = content_address(pointer)
    assert lfs.kind == "lfs_pointer"
    assert lfs.content_sha256 == oid
    assert lfs.declared_size == 42
    assert lfs.stored_sha256 == hash_bytes(pointer.read_bytes())
    assert lfs.stored_sha256 != oid

    crlf = tmp_path / "crlf.parquet"
    crlf.write_bytes(
        b"version https://git-lfs.github.com/spec/v1\r\n"
        + f"oid sha256:{oid}\r\nsize 42\r\n".encode("ascii")
    )
    crlf_address = content_address(crlf)
    assert crlf_address.kind == "lfs_pointer"
    assert crlf_address.content_sha256 == oid
    assert crlf_address.declared_size == 42


def test_concurrent_lake_publish_is_idempotent_and_leaves_no_temp_files(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    source.write_bytes(b"same immutable bytes")
    lake = tmp_path / "lake"
    with ThreadPoolExecutor(max_workers=8) as pool:
        objects = list(pool.map(lambda _: put_file(lake, source)[0], range(16)))
        snapshots = list(pool.map(lambda _: commit_snapshot(lake, "same", []), range(16)))
    assert len(set(objects)) == 1
    assert objects[0].read_bytes() == source.read_bytes()
    assert len({item.snapshot_id for item in snapshots}) == 1
    assert load_snapshot(lake, snapshots[0].snapshot_id) == snapshots[0]
    assert not list(lake.rglob("*.tmp"))


def test_partitioned_snapshot_is_immutable_and_queryable(tmp_path: Path) -> None:
    snapshot = write_partitioned_bars(tmp_path, _bars(), dataset="bars")
    again = write_partitioned_bars(tmp_path, _bars(), dataset="bars")
    assert snapshot.snapshot_id == again.snapshot_id
    assert len(snapshot.files) == 2
    for entry in snapshot.files:
        assert entry.logical_path.startswith("partitions/source=stooq/symbol=")
        assert (tmp_path / entry.object_path).is_file()
        assert content_address(tmp_path / entry.object_path).stored_sha256 == entry.stored_sha256

    visible = asof_observations(tmp_path, snapshot.snapshot_id, T1, symbols=["AAPL"])
    assert [row.symbol for row in visible] == ["AAPL", "AAPL"]
    hidden = asof_bars(tmp_path, snapshot.snapshot_id, T0)
    assert hidden.is_empty()
    with pytest.raises(PointInTimeError):
        asof_bars(tmp_path, snapshot.snapshot_id, datetime(2024, 1, 2, 14, 31))


def test_asof_rejects_knowledge_before_event(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {
            "security_id": ["A"],
            "symbol": ["AAPL"],
            "event_time": [T1],
            "available_time": [T0],
            "source": ["stooq"],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [1.0],
        }
    )
    snapshot = write_partitioned_bars(tmp_path, frame, dataset="bars")
    with pytest.raises(LeakageError):
        asof_bars(tmp_path, snapshot.snapshot_id, T1)


def test_hf_import_projects_minute_close_without_rewriting(tmp_path: Path) -> None:
    source = tmp_path / "ohlcv_2024-01.parquet"
    pl.DataFrame(
        {
            "timestamp": [T0],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [5.0],
            "ticker": ["aapl"],
        }
    ).write_parquet(source)
    raw = source.read_bytes()
    lake = tmp_path / "lake"
    snapshot = import_files([source], lake, dataset="hf", source="hf_ohlcv_1m")
    assert source.read_bytes() == raw
    stored = lake / snapshot.files[0].object_path
    assert stored.read_bytes() == raw
    assert snapshot.files[0].byte_identical_to_source is True
    assert snapshot.files[0].source == "hf_ohlcv_1m"

    assert asof_bars(lake, snapshot.snapshot_id, T0).is_empty()
    visible = asof_observations(lake, snapshot.snapshot_id, T1)
    assert len(visible) == 1
    assert visible[0].symbol == "AAPL"
    assert visible[0].event_time == T1
    assert visible[0].available_time == T1
    assert stored.read_bytes() == raw


def test_event_time_only_file_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "tape.parquet"
    pl.DataFrame(
        {
            "timestamp": [T0],
            "ticker": ["AAPL"],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
        }
    ).write_parquet(source)
    lake = tmp_path / "lake"
    snapshot = import_files([source], lake, dataset="vendor")
    with pytest.raises(DataContractError, match="knowledge-time"):
        asof_bars(lake, snapshot.snapshot_id, T1)


def test_universe_panel_drops_names_that_left(tmp_path: Path) -> None:
    path = tmp_path / "universe.parquet"
    pl.DataFrame(
        {
            "security_id": ["A", "B", "A"],
            "symbol": ["AAPL", "GONE", "AAPL"],
            "asof": [DAY, DAY, NEXT],
            "available_time": [DAY, DAY, NEXT],
            "effective_from": [DAY, DAY, DAY],
        }
    ).write_parquet(path)
    lake = tmp_path / "lake"
    snapshot = import_files([path], lake, dataset="universe")
    assert snapshot.files[0].role == "universe"
    day_one = {member.security_id for member in universe_asof(lake, snapshot.snapshot_id, DAY)}
    day_two = {member.security_id for member in universe_asof(lake, snapshot.snapshot_id, NEXT)}
    assert day_one == {"A", "B"}
    assert day_two == {"A"}


def test_interval_universe_keeps_delisted_names_until_effective_to(tmp_path: Path) -> None:
    path = tmp_path / "listings.parquet"
    later = datetime(2024, 1, 10, 21, 0, tzinfo=UTC)
    pl.DataFrame(
        {
            "security_id": ["LIVE", "DEAD", "FUTURE", "LATE"],
            "symbol": ["L", "D", "F", "X"],
            "effective_from": [DAY, DAY, later, DAY],
            "effective_to": [None, later, None, None],
            "available_time": [DAY, DAY, DAY, later],
        }
    ).write_parquet(path)
    lake = tmp_path / "lake"
    snapshot = import_files([path], lake, dataset="listings")
    members = {item.security_id for item in universe_asof(lake, snapshot.snapshot_id, NEXT)}
    assert members == {"LIVE", "DEAD"}
    after = {item.security_id for item in universe_asof(lake, snapshot.snapshot_id, later)}
    assert "DEAD" not in after
    assert "FUTURE" in after
    assert "LATE" in after


def test_static_symbol_list_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "names.parquet"
    pl.DataFrame({"symbol": ["AAPL", "MSFT"]}).write_parquet(path)
    lake = tmp_path / "lake"
    snapshot = import_files([path], lake, dataset="names", role="universe")
    with pytest.raises(DataContractError, match="survivorship-safe"):
        universe_asof(lake, snapshot.snapshot_id, DAY)


def test_membership_fallback_uses_master(tmp_path: Path) -> None:
    bars = tmp_path / "bars.parquet"
    master = tmp_path / "security_master.parquet"
    _bars().write_parquet(bars)
    pl.DataFrame(
        {
            "security_id": ["A", "B"],
            "ticker": ["AAPL", "MSFT"],
            "exchange": ["XNYS", "XNYS"],
            "security_type": ["common_stock", "common_stock"],
            "valid_from": [T0, T0],
            "valid_to": [None, None],
            "available_time": [T0, T0],
        }
    ).write_parquet(master)
    lake = tmp_path / "lake"
    entries = []
    for path, role in ((bars, "bars"), (master, "security_master")):
        destination, address = put_file(lake, path)
        entries.append(
            file_entry_for_object(
                logical_path=f"imports/{path.name}",
                object_path=destination.relative_to(lake).as_posix(),
                address=address,
                role=role,
                source="stooq",
                symbol=None,
                day=None,
                byte_identical_to_source=True,
            )
        )
    snapshot = commit_snapshot(lake, "joined", entries)
    config = UniverseConfig(
        min_price=1.0,
        min_adv=0.0,
        min_history_bars=1,
        exchanges=["XNYS"],
        security_types=["common_stock"],
        top_n_adv=None,
    )
    members = universe_asof(lake, snapshot.snapshot_id, T1, config=config)
    assert {item.security_id for item in members} == {"A", "B"}
    with pytest.raises(DataContractError, match="no PIT universe"):
        universe_asof(lake, snapshot.snapshot_id, T1)


def test_quality_fails_closed_on_each_structural_break() -> None:
    clean = quality_report(_bars())
    assert clean["passed"] is True

    dupes = pl.concat([_bars(), _bars().head(1)])
    dup_report = quality_report(dupes)
    assert dup_report["passed"] is False
    assert any(
        check["name"] == "duplicates" and not check["passed"] for check in dup_report["checks"]
    )

    backwards = _bars().select(pl.all().reverse())
    mono = quality_report(backwards)
    assert any(check["name"] == "non_monotone" and not check["passed"] for check in mono["checks"])

    broken = _bars().with_columns(pl.lit(100.0).alias("low"))
    ohlc = quality_report(broken)
    assert any(check["name"] == "ohlc" and not check["passed"] for check in ohlc["checks"])

    jump = pl.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "event_time": [T0, T1],
            "close": [10.0, 100.0],
            "open": [10.0, 100.0],
            "high": [10.0, 100.0],
            "low": [10.0, 100.0],
        }
    )
    outliers = quality_report(jump, QualityThresholds(max_abs_log_return=0.5))
    assert any(check["name"] == "outliers" and not check["passed"] for check in outliers["checks"])

    stale_rows = pl.DataFrame(
        {
            "symbol": ["AAPL"] * 5,
            "event_time": [datetime(2024, 1, day, 21, tzinfo=UTC) for day in range(2, 7)],
            "close": [10.0] * 5,
            "open": [10.0] * 5,
            "high": [10.0] * 5,
            "low": [10.0] * 5,
        }
    )
    stale = quality_report(stale_rows, QualityThresholds(stale_run_length=5, max_stale_runs=0))
    assert any(check["name"] == "stale_prices" and not check["passed"] for check in stale["checks"])

    gaps = quality_report(stale_rows, QualityThresholds(max_gap=timedelta(hours=12), max_gaps=0))
    assert any(check["name"] == "gaps" and not check["passed"] for check in gaps["checks"])
    with pytest.raises(QualityThresholdError):
        quality_report(broken, enforce=True)


def test_lfs_pointer_quality_fails_closed(tmp_path: Path) -> None:
    pointer = tmp_path / "features.parquet"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:" + ("cd" * 32) + "\nsize 9\n",
        encoding="ascii",
    )
    report = quality_report_path(pointer, enforce=False)
    assert report["passed"] is False
    assert report["kind"] == "lfs_pointer"
    with pytest.raises(QualityThresholdError):
        quality_report_path(pointer, enforce=True)


def test_migration_preserves_research_output_bytes(tmp_path: Path) -> None:
    source = tmp_path / "bronze.parquet"
    _bars().write_parquet(source)
    before_bytes = source.read_bytes()
    before_panel = lagged_return_panel_bytes(source)
    lake = tmp_path / "lake"
    snapshot = import_files([source], lake, dataset="bronze")
    assert source.read_bytes() == before_bytes
    stored = lake / snapshot.files[0].object_path
    assert stored.read_bytes() == before_bytes
    assert lagged_return_panel_bytes(stored) == before_panel
    reloaded = load_snapshot(lake, snapshot.snapshot_id)
    assert reloaded.snapshot_id == snapshot.snapshot_id
    fields = snapshot_receipt_fields(snapshot.snapshot_id)
    assert FORBIDDEN_RESEARCH_METRIC_KEYS.isdisjoint(fields)
    assert fields["data_snapshot_id"] == snapshot.snapshot_id


def test_lineage_dag_and_verify_detects_drift(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    bars = write_partitioned_bars(lake, _bars(), dataset="bars")
    features = write_partitioned_bars(lake, _bars(), dataset="features")
    code = tmp_path / "producer.py"
    code.write_text("def build():\n    return 1\n", encoding="utf-8")
    record_lineage(
        lake,
        dataset="features",
        snapshot_id=features.snapshot_id,
        inputs=(LineageInput("bars", bars.snapshot_id),),
        code=code,
        parameters={"window": 2},
    )
    record_lineage(
        lake,
        dataset="bars",
        snapshot_id=bars.snapshot_id,
        code="root-import",
        parameters={},
    )
    text = format_dag(lake, "features")
    assert "features" in text
    assert "bars" in text
    assert features.snapshot_id in text
    assert bars.snapshot_id in text
    assert verify_lineage(lake, "features").ok is True

    object_path = lake / features.files[0].object_path
    original = object_path.read_bytes()
    object_path.write_bytes(original + b"\x00")
    drifted = verify_lineage(lake, "features")
    assert drifted.ok is False
    assert any(item.startswith("stored_hash_drift:") for item in drifted.errors)


def test_code_hash_drift_after_clean_objects(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    bars = write_partitioned_bars(lake, _bars(), dataset="bars")
    code = tmp_path / "producer.py"
    code.write_text("def build():\n    return 1\n", encoding="utf-8")
    record_lineage(
        lake,
        dataset="bars",
        snapshot_id=bars.snapshot_id,
        code=code,
        parameters={"window": 2},
    )
    assert verify_lineage(lake).ok is True
    code.write_text("def build():\n    return 2\n", encoding="utf-8")
    report = verify_lineage(lake, "bars")
    assert report.ok is False
    assert any(item.startswith("code_hash_drift:") for item in report.errors)


def test_cli_show_and_verify(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    bars = write_partitioned_bars(lake, _bars(), dataset="bars")
    record_lineage(
        lake,
        dataset="bars",
        snapshot_id=bars.snapshot_id,
        code="import",
        parameters={},
    )
    runner = CliRunner()
    shown = runner.invoke(app, ["lineage", "show", "bars", "--root", str(lake)])
    assert shown.exit_code == 0
    assert bars.snapshot_id in shown.stdout
    verified = runner.invoke(app, ["lineage", "verify", "bars", "--root", str(lake)])
    assert verified.exit_code == 0
    assert json.loads(verified.stdout)["ok"] is True
    missing = runner.invoke(app, ["lineage", "show", "missing", "--root", str(lake)])
    assert missing.exit_code == 1


def test_snapshot_tamper_fails_closed(tmp_path: Path) -> None:
    snapshot = write_partitioned_bars(tmp_path, _bars(), dataset="bars")
    path = tmp_path / "snapshots" / f"{snapshot.snapshot_id}.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    body["dataset"] = "other"
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(DataContractError, match="does not match"):
        load_snapshot(tmp_path, snapshot.snapshot_id)


def test_receipt_fields_reject_a_bad_id() -> None:
    with pytest.raises(DataContractError):
        snapshot_receipt_fields("not-a-hash")
