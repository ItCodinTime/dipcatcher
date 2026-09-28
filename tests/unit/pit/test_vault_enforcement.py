"""PIT vault enforcement hardening: audit-pinned invariants and regressions.

Covers the findings in docs/AUDIT_PIT_VAULT.md: dataset.json anchoring,
manifest shape checks, dataset nesting, symlink handling, duplicate-version
rejection, the LH009 history() gate, and lock-error taxonomy — plus the
pre-existing invariants the audit re-pinned (out-of-order revision rejection,
forged manifest hashes, symlinked vault paths, asof boundary inclusivity).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from typer.testing import CliRunner

from quant_fund.leakage import scan_paths
from quant_fund.pit import (
    ManifestError,
    PitVault,
    VaultError,
    VaultUnavailableError,
)
from quant_fund.pit import manifest as manifest_mod
from quant_fund.pit.cli import pit_app
from quant_fund.pit.manifest import read_manifest, sha256_file
from quant_fund.pit.shim import guarded_read_parquet
from quant_fund.proofcore.contracts import (
    GENESIS_HASH,
    PitManifest,
)
from quant_fund.schemas.errors import PointInTimeError

T0 = datetime(2024, 1, 1, tzinfo=UTC)
runner = CliRunner()


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


def _manifest_payload(root, name: str) -> dict:
    return json.loads(manifest_mod.manifest_path(root, name).read_bytes())


def _rewrite_head(root, name: str, payload: dict) -> None:
    """Write ``payload`` as both the retained head revision and the pointer."""
    manifest = PitManifest.model_validate(payload)
    payload_bytes = manifest_mod.manifest_json_bytes(manifest)
    manifest_mod.manifest_path(root, name).write_bytes(payload_bytes)
    manifest_mod.versioned_manifest_path(root, name, manifest.revision).write_bytes(payload_bytes)


# -- dataset.json anchoring (create-time immutable metadata) ----------------


def test_dataset_meta_anchor_detects_bit_flip(vault: PitVault, tmp_path) -> None:
    """Tampering with dataset.json (e.g. security_level false) is caught."""
    vault.append("silver/bars", _frame([("A", 0, 0, 1.0)]))
    meta_path = tmp_path / "pit" / "silver" / "bars" / "dataset.json"
    meta = json.loads(meta_path.read_text())
    meta["security_level"] = False
    meta_path.write_text(json.dumps(meta, sort_keys=True))
    with pytest.raises(ManifestError, match="dataset.json sha256 disagrees"):
        vault.asof("silver/bars", T0 + timedelta(days=1))
    assert vault.verify("silver/bars") != []


def test_dataset_meta_anchor_detects_deletion(vault: PitVault, tmp_path) -> None:
    vault.append("silver/bars", _frame([("A", 0, 0, 1.0)]))
    (tmp_path / "pit" / "silver" / "bars" / "dataset.json").unlink()
    with pytest.raises(VaultError, match="unknown dataset"):
        vault.asof("silver/bars", T0 + timedelta(days=1))
    assert any("dataset.json missing" in v for v in vault.verify("silver/bars"))


def test_dataset_meta_name_field_must_match(vault: PitVault, tmp_path) -> None:
    """A foreign dataset.json (swapped in whole) is rejected on the name field."""
    vault.create_dataset("gold/macro")
    bars_meta = tmp_path / "pit" / "silver" / "bars" / "dataset.json"
    macro_meta = tmp_path / "pit" / "gold" / "macro" / "dataset.json"
    macro_meta.write_bytes(bars_meta.read_bytes())
    with pytest.raises(VaultError, match="dataset.json malformed"):
        vault.append("gold/macro", _frame([("A", 0, 0, 1.0)]))


def test_dataset_meta_anchor_carried_across_appends(vault: PitVault) -> None:
    vault.append("silver/bars", _frame([("A", 0, 0, 1.0)]))
    vault.append("silver/bars", _frame([("A", 1, 1, 2.0)]))
    manifest = read_manifest(vault.root, "silver/bars")
    genesis = json.loads(
        manifest_mod.versioned_manifest_path(vault.root, "silver/bars", 0).read_bytes()
    )
    assert manifest.dataset_meta_sha256 == genesis["dataset_meta_sha256"]
    assert manifest.dataset_meta_sha256 != GENESIS_HASH


def test_legacy_unanchored_manifest_still_reads(vault: PitVault) -> None:
    """Pre-anchor manifests (no dataset_meta_sha256 field) stay readable."""
    payload = _manifest_payload(vault.root, "silver/bars")
    del payload["dataset_meta_sha256"]
    _rewrite_head(vault.root, "silver/bars", payload)
    vault.append("silver/bars", _frame([("A", 0, 0, 1.0)]))
    frame = vault.asof("silver/bars", T0 + timedelta(days=1))
    assert frame.rows == 1
    assert read_manifest(vault.root, "silver/bars").dataset_meta_sha256 == GENESIS_HASH


# -- manifest structural shape -----------------------------------------------


def test_forge_rev0_listing_files_rejected(vault: PitVault) -> None:
    """A consistent rev-0 manifest claiming files fails the shape check."""
    payload = _manifest_payload(vault.root, "silver/bars")
    payload["files"] = [
        {
            "path": "silver/bars/parts/r0000001.parquet",
            "sha256": "a" * 64,
            "rows": 1,
            "min_known_at": T0.isoformat(),
            "max_known_at": T0.isoformat(),
            "min_event_time": T0.isoformat(),
            "max_event_time": T0.isoformat(),
        }
    ]
    _rewrite_head(vault.root, "silver/bars", payload)
    with pytest.raises(ManifestError, match="revision 0 lists 1 files"):
        vault.asof("silver/bars", T0 + timedelta(days=1))


def test_forge_nonsequential_part_path_rejected(vault: PitVault) -> None:
    """A manifest entry pointing at r0000042 instead of r0000001 is forged."""
    vault.append("silver/bars", _frame([("A", 0, 0, 1.0)]))
    payload = _manifest_payload(vault.root, "silver/bars")
    payload["files"][0]["path"] = "silver/bars/parts/r0000042.parquet"
    _rewrite_head(vault.root, "silver/bars", payload)
    with pytest.raises(ManifestError, match="file 1 path .* != expected"):
        vault.asof("silver/bars", T0 + timedelta(days=1))


def test_metadata_anchor_drift_across_revisions_rejected(vault: PitVault) -> None:
    """A retained revision with a different metadata anchor fails closed."""
    vault.append("silver/bars", _frame([("A", 0, 0, 1.0)]))
    vault.append("silver/bars", _frame([("A", 1, 1, 2.0)]))
    r1_path = manifest_mod.versioned_manifest_path(vault.root, "silver/bars", 1)
    payload = json.loads(r1_path.read_bytes())
    payload["dataset_meta_sha256"] = "b" * 64
    r1_path.write_bytes(manifest_mod.manifest_json_bytes(PitManifest.model_validate(payload)))
    with pytest.raises(ManifestError, match="metadata anchor drift"):
        vault.asof("silver/bars", T0 + timedelta(days=2))


# -- pre-existing invariants, pinned -----------------------------------------


def test_out_of_order_revision_rejected(vault: PitVault) -> None:
    """A retained revision whose prev link is wrong breaks the chain."""
    vault.append("silver/bars", _frame([("A", 0, 0, 1.0)]))
    vault.append("silver/bars", _frame([("A", 1, 1, 2.0)]))
    r1_path = manifest_mod.versioned_manifest_path(vault.root, "silver/bars", 1)
    payload = json.loads(r1_path.read_bytes())
    payload["prev_manifest_sha256"] = GENESIS_HASH  # lies: should be sha(r0)
    r1_path.write_bytes(manifest_mod.manifest_json_bytes(PitManifest.model_validate(payload)))
    with pytest.raises(ManifestError, match="manifest chain broken at revision 1"):
        vault.asof("silver/bars", T0 + timedelta(days=2))


def test_forged_manifest_file_hash_rejected(vault: PitVault) -> None:
    """A manifest entry with a forged part sha256 fails part verification."""
    vault.append("silver/bars", _frame([("A", 0, 0, 1.0)]))
    payload = _manifest_payload(vault.root, "silver/bars")
    payload["files"][0]["sha256"] = "f" * 64
    _rewrite_head(vault.root, "silver/bars", payload)
    with pytest.raises(ManifestError, match="committed part sha256 mismatch"):
        vault.asof("silver/bars", T0 + timedelta(days=1))
    assert any("sha256 mismatch" in v for v in vault.verify("silver/bars"))


def test_asof_boundary_inclusive(vault: PitVault) -> None:
    """known_at == t is returned; nothing exists with all known_at > t."""
    vault.append("silver/bars", _frame([("A", 0, 2, 1.0)]))
    assert vault.asof("silver/bars", T0 + timedelta(days=2)).rows == 1
    with pytest.raises(VaultUnavailableError):
        vault.asof("silver/bars", T0 + timedelta(days=1))


def test_symlinked_dataset_dir_outside_root_rejected(tmp_path) -> None:
    """A dataset dir symlinked outside the vault root refuses before reading."""
    vault = PitVault(tmp_path / "pit")
    outside = tmp_path / "outside"
    outside.mkdir()
    vault.root.mkdir(parents=True)
    (vault.root / "escape").symlink_to(outside)
    with pytest.raises(VaultError, match="escapes vault root"):
        vault.create_dataset("escape")
    with pytest.raises(VaultError, match="escapes vault root"):
        vault.asof("escape", T0)


# -- dataset nesting ----------------------------------------------------------


def test_nested_dataset_names_rejected(vault: PitVault) -> None:
    """A dataset may not nest under or over an existing one."""
    with pytest.raises(VaultError, match="must not nest"):
        vault.create_dataset("silver/bars/deep")
    with pytest.raises(VaultError, match="must not nest"):
        vault.create_dataset("silver")
    vault.create_dataset("silver/other")  # sibling is fine


# -- dangling symlinks & uncommitted part recovery -----------------------------


def test_dangling_symlink_wedges_append_and_fails_loud_on_recover(
    vault: PitVault, tmp_path
) -> None:
    vault.append("silver/bars", _frame([("A", 0, 0, 1.0)]))
    orphan = tmp_path / "pit" / "silver" / "bars" / "parts" / "r0000002.parquet"
    orphan.symlink_to(tmp_path / "nonexistent-target")
    with pytest.raises(VaultError, match="write-once violation"):
        vault.append("silver/bars", _frame([("B", 0, 0, 2.0)]))
    with pytest.raises(VaultError, match="unsafe uncommitted part"):
        vault.recover_uncommitted_part("silver/bars")
    orphan.unlink()  # operator inspection decides: it is junk, remove it
    vault.append("silver/bars", _frame([("B", 0, 0, 2.0)]))
    assert vault.asof("silver/bars", T0 + timedelta(days=1)).rows == 2


def test_recover_uncommitted_part_clean_path(vault: PitVault, tmp_path) -> None:
    assert vault.recover_uncommitted_part("silver/bars") is False
    vault.append("silver/bars", _frame([("A", 0, 0, 1.0)]))
    orphan = tmp_path / "pit" / "silver" / "bars" / "parts" / "r0000002.parquet"
    orphan.write_bytes(b"interrupted")
    assert vault.recover_uncommitted_part("silver/bars") is True
    vault.append("silver/bars", _frame([("B", 0, 0, 2.0)]))
    assert sha256_file(orphan) != sha256_file(
        tmp_path / "pit" / "silver" / "bars" / "parts" / "r0000001.parquet"
    )


def test_directory_at_part_path_fails_loud(vault: PitVault, tmp_path) -> None:
    """A directory where the next part must land is unsafe, not silent."""
    vault.append("silver/bars", _frame([("A", 0, 0, 1.0)]))
    (tmp_path / "pit" / "silver" / "bars" / "parts" / "r0000002.parquet").mkdir()
    with pytest.raises(VaultError, match="write-once violation"):
        vault.append("silver/bars", _frame([("B", 0, 0, 2.0)]))
    with pytest.raises(VaultError, match="unsafe uncommitted part"):
        vault.recover_uncommitted_part("silver/bars")


# -- duplicate versions --------------------------------------------------------


def test_duplicate_key_known_at_append_refused(vault: PitVault) -> None:
    """Two versions of one (key, known_at) in a single append are ambiguous."""
    with pytest.raises(VaultError, match="duplicate .*known_at"):
        vault.append("silver/bars", _frame([("A", 0, 0, 1.0), ("A", 0, 0, 2.0)]))
    vault.append("silver/bars", _frame([("A", 0, 0, 1.0), ("A", 0, 1, 2.0)]))
    assert vault.asof("silver/bars", T0 + timedelta(days=2)).rows == 1


# -- append lock error taxonomy -------------------------------------------------


def test_append_lock_unavailable_maps_to_vault_error(
    vault: PitVault, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fail(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr("quant_fund.pit.vault.sqlite3.connect", _fail)
    with pytest.raises(VaultError, match="append lock unavailable"):
        vault.append("silver/bars", _frame([("A", 0, 0, 1.0)]))


# -- LH009: history() is audit-only ---------------------------------------------


def test_lh009_flags_history_call(tmp_path) -> None:
    report = scan_paths([_src(tmp_path, "return vault.history(name, key)\n")])
    assert any(f.rule_id == "LH009" for f in report.findings)


def test_lh009_ignores_asof_call(tmp_path) -> None:
    report = scan_paths([_src(tmp_path, "return vault.asof(name, t)\n")])
    assert not any(f.rule_id == "LH009" for f in report.findings)


def _src(tmp_path, body: str):
    path = tmp_path / "mod.py"
    path.write_text(f"def f(vault, name, key, t):\n    {body}", encoding="utf-8")
    return path


# -- CLI stats tolerates corrupt datasets ----------------------------------------


def test_pit_stats_continues_past_corrupt_manifest(tmp_path) -> None:
    root = tmp_path / "pit"
    vault = PitVault(root)
    vault.create_dataset("silver/bars")
    vault.append("silver/bars", _frame([("A", 0, 0, 1.0)]))
    vault.create_dataset("gold/macro")
    manifest_mod.manifest_path(root, "silver/bars").write_bytes(b"{}")
    result = runner.invoke(pit_app, ["stats", "--root", str(root)])
    assert result.exit_code == 0, result.output
    assert "manifest unreadable" in result.output
    assert "gold/macro" in result.output


# -- shim: event_time dtype gate ---------------------------------------------------


def test_shim_refuses_naive_event_time(tmp_path) -> None:
    """Legacy frames with naive event_time fail closed like naive known_at."""
    frame = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [datetime(2024, 1, 1)],  # naive
            "known_at": [T0],
            "close": [1.0],
        }
    )
    part = tmp_path / "legacy.parquet"
    frame.write_parquet(part)
    with pytest.raises(PointInTimeError, match="event_time"):
        guarded_read_parquet(part, T0 + timedelta(days=1), dataset="legacy")
