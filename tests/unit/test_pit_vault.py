"""Unit tests for PitVault: storage format, manifest chain, write-once, asof.

DESIGN.md §12 W1 row: manifest chain, write-once refusal, corrupted part
bytes -> verify() flags, manifest tamper -> ManifestError.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_fund.pit import (
    ManifestError,
    PitFrame,
    PitVault,
    RestatementPolicy,
    VaultError,
    VaultUnavailableError,
)
from quant_fund.pit.manifest import read_manifest, sha256_file
from quant_fund.schemas.errors import PointInTimeError

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _frame(rows: list[tuple[str, int, int, float]]) -> pl.DataFrame:
    """rows: (security_id, event_day_offset, known_day_offset, value)."""
    return pl.DataFrame(
        {
            "security_id": [r[0] for r in rows],
            "event_time": [T0 + timedelta(days=r[1]) for r in rows],
            "known_at": [T0 + timedelta(days=r[2]) for r in rows],
            "close": [r[3] for r in rows],
        }
    )


@pytest.fixture()
def vault(tmp_path) -> PitVault:
    v = PitVault(tmp_path / "pit")
    v.create_dataset("silver/bars")
    return v


def test_create_dataset_layout(vault: PitVault, tmp_path) -> None:
    ds = tmp_path / "pit" / "silver" / "bars"
    assert (ds / "parts").is_dir()
    assert (ds / "manifest.json").is_file()
    assert (ds / "manifest.sha256").is_file()
    manifest = read_manifest(vault.root, "silver/bars")
    assert manifest.revision == 0
    assert manifest.files == []
    assert manifest.prev_manifest_sha256 == "0" * 64
    assert (ds / "manifest.sha256").read_text().strip() == "0" * 64


def test_create_dataset_twice_refused(vault: PitVault) -> None:
    with pytest.raises(VaultError, match="already exists"):
        vault.create_dataset("silver/bars")


def test_create_dataset_illegal_name(vault: PitVault) -> None:
    for bad in ("../escape", "/abs", "a\\b", ""):
        with pytest.raises(VaultError):
            vault.create_dataset(bad)


def test_append_writes_part_and_manifest(vault: PitVault) -> None:
    manifest = vault.append("silver/bars", _frame([("A", 0, 0, 100.0)]))
    assert manifest.revision == 1
    assert len(manifest.files) == 1
    entry = manifest.files[0]
    assert entry.path == "silver/bars/parts/r0000001.parquet"
    assert entry.rows == 1
    part = vault.root / entry.path
    assert part.is_file()
    assert entry.sha256 == sha256_file(part)
    assert vault.verify("silver/bars") == []


def test_manifest_chain_links(vault: PitVault) -> None:
    import hashlib

    vault.append("silver/bars", _frame([("A", 0, 0, 100.0)]))
    first_bytes = (vault.root / "silver/bars/manifest.json").read_bytes()
    first_sha = hashlib.sha256(first_bytes).hexdigest()
    second = vault.append("silver/bars", _frame([("B", 1, 1, 50.0)]))
    assert second.revision == 2
    assert second.prev_manifest_sha256 == first_sha
    sidecar = (vault.root / "silver/bars/manifest.sha256").read_text().strip()
    assert sidecar == first_sha


def test_manifest_tamper_raises(vault: PitVault) -> None:
    vault.append("silver/bars", _frame([("A", 0, 0, 100.0)]))
    sidecar = vault.root / "silver/bars/manifest.sha256"
    sidecar.write_text("f" * 64 + "\n")
    with pytest.raises(ManifestError, match="chain broken"):
        read_manifest(vault.root, "silver/bars")
    with pytest.raises(ManifestError):
        vault.asof("silver/bars", T0 + timedelta(days=1))


def test_manifest_malformed_json_raises(vault: PitVault) -> None:
    (vault.root / "silver/bars/manifest.json").write_bytes(b'{"dataset":')
    with pytest.raises(ManifestError, match="malformed"):
        read_manifest(vault.root, "silver/bars")


def test_corrupted_part_flagged_by_verify(vault: PitVault) -> None:
    vault.append("silver/bars", _frame([("A", 0, 0, 100.0)]))
    part = vault.root / "silver/bars/parts/r0000001.parquet"
    with open(part, "ab") as handle:
        handle.write(b"bitrot")
    violations = vault.verify("silver/bars")
    assert any("sha256 mismatch" in v for v in violations)


def test_unlisted_part_flagged_by_verify(vault: PitVault) -> None:
    vault.append("silver/bars", _frame([("A", 0, 0, 100.0)]))
    _frame([("Z", 9, 9, 1.0)]).write_parquet(vault.root / "silver/bars/parts/r9999999.parquet")
    violations = vault.verify("silver/bars")
    assert any("not listed in manifest" in v for v in violations)


def test_write_once_refusal(vault: PitVault) -> None:
    vault.append("silver/bars", _frame([("A", 0, 0, 100.0)]))
    # Adversary pre-creates the next part name; append must refuse to overwrite.
    _frame([("X", 5, 5, 999.0)]).write_parquet(vault.root / "silver/bars/parts/r0000002.parquet")
    with pytest.raises(VaultError, match="write-once"):
        vault.append("silver/bars", _frame([("B", 1, 1, 50.0)]))
    survivor = pl.read_parquet(vault.root / "silver/bars/parts/r0000002.parquet")
    assert survivor["security_id"].to_list() == ["X"]


def test_append_validation_fail_closed(vault: PitVault) -> None:
    with pytest.raises(VaultError, match="missing PIT columns"):
        vault.append("silver/bars", pl.DataFrame({"security_id": ["A"], "close": [1.0]}))
    with pytest.raises(VaultError, match="timezone-aware"):
        vault.append(
            "silver/bars",
            pl.DataFrame(
                {
                    "security_id": ["A"],
                    "event_time": [datetime(2024, 1, 1)],  # naive
                    "known_at": [T0],
                    "close": [1.0],
                }
            ),
        )
    with pytest.raises(VaultError, match="empty append"):
        vault.append(
            "silver/bars",
            _frame([("A", 0, 0, 1.0)]).filter(pl.lit(False)),
        )
    with pytest.raises(VaultError, match="security_id"):
        vault.append(
            "silver/bars",
            pl.DataFrame({"event_time": [T0], "known_at": [T0], "close": [1.0]}),
        )
    with pytest.raises(VaultError, match="null"):
        vault.append(
            "silver/bars",
            pl.DataFrame(
                {
                    "security_id": ["A", "B"],
                    "event_time": [T0, None],
                    "known_at": [T0, T0],
                    "close": [1.0, 2.0],
                }
            ),
        )


def test_append_unknown_dataset_refused(tmp_path) -> None:
    vault = PitVault(tmp_path / "pit")
    with pytest.raises(VaultError, match="unknown dataset"):
        vault.append("nope", _frame([("A", 0, 0, 1.0)]))


def test_asof_latest_known(vault: PitVault) -> None:
    vault.append("silver/bars", _frame([("A", 0, 0, 100.0), ("B", 0, 0, 50.0)]))
    vault.restate(
        "silver/bars",
        pl.DataFrame({"security_id": ["A"], "event_time": [T0], "close": [101.5]}),
        known_at=T0 + timedelta(days=5),
    )
    early = vault.asof("silver/bars", T0 + timedelta(days=1))
    assert early.frame.filter(pl.col("security_id") == "A")["close"].to_list() == [100.0]
    late = vault.asof("silver/bars", T0 + timedelta(days=6))
    assert late.frame.filter(pl.col("security_id") == "A")["close"].to_list() == [101.5]
    assert late.max_known_at == T0 + timedelta(days=5)


def test_asof_unavailable_raises_and_maps_to_point_in_time(vault: PitVault) -> None:
    vault.append("silver/bars", _frame([("A", 0, 5, 100.0)]))
    with pytest.raises(VaultUnavailableError):
        vault.asof("silver/bars", T0 + timedelta(days=1))
    # §4.3 boundary mapping: existing `except PointInTimeError` callers catch it.
    with pytest.raises(PointInTimeError):
        vault.asof("silver/bars", T0 + timedelta(days=1))


def test_asof_empty_dataset_unavailable(vault: PitVault) -> None:
    with pytest.raises(VaultUnavailableError):
        vault.asof("silver/bars", T0)


def test_asof_naive_timestamp_refused(vault: PitVault) -> None:
    vault.append("silver/bars", _frame([("A", 0, 0, 100.0)]))
    with pytest.raises(VaultError, match="timezone-aware"):
        vault.asof("silver/bars", datetime(2024, 1, 2))


def test_asof_unknown_dataset(vault: PitVault) -> None:
    with pytest.raises(VaultError, match="unknown dataset"):
        vault.asof("nope", T0)


def test_asof_columns_projection(vault: PitVault) -> None:
    frame = _frame([("A", 0, 0, 100.0)]).with_columns(pl.lit(7.0).alias("volume"))
    vault.append("silver/bars", frame)
    out = vault.asof("silver/bars", T0, columns=["close"])
    assert out.frame.columns == ["security_id", "event_time", "known_at", "close"]
    with pytest.raises(VaultError, match="unknown column"):
        vault.asof("silver/bars", T0, columns=["does_not_exist"])


def test_asof_records_into_recorder_and_watchdog(vault: PitVault) -> None:
    recorded: list = []
    observed: list = []

    class _Recorder:
        def record(self, read) -> None:
            recorded.append(read)

    class _Watchdog:
        def observe(self, read, decision_time) -> None:
            observed.append((read, decision_time))

    vault = PitVault(vault.root, recorder=_Recorder(), watchdog=_Watchdog())
    vault.append("silver/bars", _frame([("A", 0, 0, 100.0)]))
    out = vault.asof("silver/bars", T0, columns=["close"])
    assert len(recorded) == 1
    read = recorded[0]
    assert read.dataset == "silver/bars"
    assert read.rows == out.rows == 1
    assert read.content_sha256 == out.content_sha256
    assert len(read.content_sha256) == 64
    assert read.params == {"policy": "latest_known", "columns": "close"}
    assert observed == [(read, T0)]


def test_history_returns_all_versions(vault: PitVault) -> None:
    vault.append("silver/bars", _frame([("A", 0, 0, 100.0)]))
    vault.restate(
        "silver/bars",
        pl.DataFrame({"security_id": ["A"], "event_time": [T0], "close": [101.5]}),
        known_at=T0 + timedelta(days=5),
    )
    hist = vault.history("silver/bars", ("A", T0))
    assert hist.height == 2
    assert hist["close"].to_list() == [100.0, 101.5]
    assert hist["known_at"].to_list() == sorted(hist["known_at"].to_list())


def test_pitframe_validate_fail_closed() -> None:
    future = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [T0],
            "known_at": [T0 + timedelta(days=10)],
            "close": [1.0],
        }
    )
    pf = PitFrame.build(future, dataset="d", asof=T0 + timedelta(days=10))
    with pytest.raises(VaultUnavailableError):
        pf.validate(T0)  # decision_time before max known_at: refuse
    with pytest.raises(VaultError):
        pf.validate(datetime(2024, 1, 1))  # naive decision_time: refuse
    pf.validate(T0 + timedelta(days=10))  # ok


def test_non_security_level_dataset(tmp_path) -> None:
    vault = PitVault(tmp_path / "pit")
    vault.create_dataset("gold/macro", security_level=False)
    with pytest.raises(VaultError, match="missing PIT columns"):
        vault.append("gold/macro", pl.DataFrame({"close": [1.0]}))
    vault.append(
        "gold/macro",
        pl.DataFrame({"event_time": [T0], "known_at": [T0], "cpi": [3.1]}),
    )
    vault.restate(
        "gold/macro",
        pl.DataFrame({"event_time": [T0], "cpi": [3.2]}),
        known_at=T0 + timedelta(days=3),
    )
    early = vault.asof("gold/macro", T0 + timedelta(days=1))
    assert early.frame["cpi"].to_list() == [3.1]
    late = vault.asof("gold/macro", T0 + timedelta(days=4))
    assert late.frame["cpi"].to_list() == [3.2]
    hist = vault.history("gold/macro", ("ignored", T0))
    assert hist.height == 2


def test_strict_first_policy(vault: PitVault) -> None:
    vault.append("silver/bars", _frame([("A", 0, 0, 100.0)]))
    vault.restate(
        "silver/bars",
        pl.DataFrame({"security_id": ["A"], "event_time": [T0], "close": [101.5]}),
        known_at=T0 + timedelta(days=5),
    )
    first = vault.asof("silver/bars", T0 + timedelta(days=6), policy=RestatementPolicy.STRICT_FIRST)
    assert first.frame["close"].to_list() == [100.0]
    latest = vault.asof("silver/bars", T0 + timedelta(days=6))
    assert latest.frame["close"].to_list() == [101.5]
