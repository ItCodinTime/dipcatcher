"""Evidence chain-of-custody tests (ULTRAPLAN P7.7).

- per-file verification composes verify_receipt_file over a directory
- set-level checks: filename↔digest binding, duplicate seals, unsealed
  accounting, unparseable files, stale evidence index
- the audit emits a sealed receipt.v2 of kind evidence_audit
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.research.evidence_audit import (
    audit_receipts_dir,
    check_evidence_index_fresh,
    evidence_audit_contract_errors,
    format_evidence_audit_table,
    run_evidence_audit,
    write_evidence_audit_receipt,
)
from quant_fund.research.receipt_v2 import (
    build_receipt_v2,
    seal_receipt,
    verify_receipt_file,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _sealed_receipt(value: int = 1) -> dict:
    receipt = build_receipt_v2(
        kind="unit_probe",
        data_label="SYNTHETIC",
        dataset={"x": "probe"},
        params={"n": 1},
        code_files=(Path(__file__),),
        verdict="pass",
        payload={"schema": "probe.v1", "live_pnl_claim": False, "value": value},
    )
    return seal_receipt(receipt)


def _write(path: Path, body: dict) -> Path:
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return path


def test_audit_clean_dir_all_sealed(tmp_path: Path) -> None:
    _write(tmp_path / "a.json", _sealed_receipt(1))
    _write(tmp_path / "b.json", _sealed_receipt(2))
    rows, receipt = run_evidence_audit(tmp_path, check_index=False)
    assert [r["file"] for r in rows] == ["a.json", "b.json"]
    assert all(r["valid"] and r["sealed"] for r in rows)
    assert receipt["verdict"] == "pass"
    assert receipt["payload"]["findings"] == []
    assert receipt["payload"]["n_files"] == 2
    assert receipt["payload"]["n_sealed"] == 2
    assert receipt["kind"] == "evidence_audit"
    assert receipt["data_label"] == "META"


def test_unsealed_reported_not_failed(tmp_path: Path) -> None:
    _write(tmp_path / "sealed_abc.json", _sealed_receipt())
    _write(tmp_path / "legacy.json", {"schema": "legacy.v1", "a": 1})
    rows, receipt = run_evidence_audit(tmp_path, check_index=False)
    legacy = next(r for r in rows if r["file"] == "legacy.json")
    assert legacy["sealed"] is False and legacy["valid"] is False
    assert receipt["verdict"] == "pass"  # unsealed is a coverage gap, not a violation
    assert receipt["payload"]["n_unsealed"] == 1


def test_tampered_seal_is_hard_finding(tmp_path: Path) -> None:
    bad = _sealed_receipt()
    bad["payload"]["value"] = 999
    _write(tmp_path / "tampered.json", bad)
    rows, receipt = run_evidence_audit(tmp_path, check_index=False)
    assert rows[0]["valid"] is False and rows[0]["sealed"] is True
    assert receipt["verdict"] == "fail"
    assert any("sealed_receipt_invalid" in f for f in receipt["payload"]["findings"])


def test_filename_digest_binding(tmp_path: Path) -> None:
    sealed = _sealed_receipt()
    good_name = f"probe_{sealed['receipt_sha256'][:16]}.json"
    rows, receipt = run_evidence_audit(
        _write_dir(tmp_path / "ok", good_name, sealed), check_index=False
    )
    assert rows[0]["filename_digest_ok"] is True
    assert receipt["verdict"] == "pass"

    bad_dir = tmp_path / "bad"
    _write_dir(bad_dir, "probe_0000000000000000.json", sealed)
    rows, receipt = run_evidence_audit(bad_dir, check_index=False)
    assert rows[0]["filename_digest_ok"] is False
    assert receipt["verdict"] == "fail"
    assert any("filename_digest_mismatch" in f for f in receipt["payload"]["findings"])


def _write_dir(directory: Path, name: str, body: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    _write(directory / name, body)
    return directory


def test_duplicate_seals_flagged(tmp_path: Path) -> None:
    sealed = _sealed_receipt()
    _write(tmp_path / "copy_a.json", sealed)
    _write(tmp_path / "copy_b.json", sealed)
    rows, receipt = run_evidence_audit(tmp_path, check_index=False)
    assert all(r["valid"] for r in rows)
    assert receipt["verdict"] == "fail"
    dup = receipt["payload"]["duplicate_seals"]
    assert len(dup) == 1
    assert sorted(next(iter(dup.values()))) == ["copy_a.json", "copy_b.json"]


def test_unparseable_and_empty_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text("{ not json")
    rows, receipt = run_evidence_audit(tmp_path, check_index=False)
    assert rows[0]["valid"] is False
    assert any("unparseable" in f for f in receipt["payload"]["findings"])

    empty = tmp_path / "empty"
    empty.mkdir()
    rows, receipt = run_evidence_audit(empty, check_index=False)
    assert rows == []
    assert receipt["payload"]["findings"] == ["no_receipts_found"]
    assert receipt["verdict"] == "fail"


def test_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="receipts directory not found"):
        run_evidence_audit(tmp_path / "nope", check_index=False)


def test_contract_errors_fail_closed() -> None:
    receipt = seal_receipt(
        build_receipt_v2(
            kind="evidence_audit",
            data_label="META",
            dataset={},
            params={},
            code_files=(Path(__file__),),
            verdict="pass",
            payload={
                "schema": "evidence_audit.v1",
                "live_pnl_claim": False,
                "files": [],
                "n_files": 0,
            },
        )
    )
    assert evidence_audit_contract_errors(receipt) == []
    bad = dict(receipt)
    bad["kind"] = "other"
    assert "kind_mismatch" in evidence_audit_contract_errors(bad)
    assert "receipt_not_object" in evidence_audit_contract_errors("x")


def test_write_uses_digest_filename(tmp_path: Path) -> None:
    _, receipt = run_evidence_audit(tmp_path, check_index=False)
    path = write_evidence_audit_receipt(receipt, tmp_path)
    assert path.name == f"evidence_audit_{receipt['receipt_sha256'][:16]}.json"
    # and the audit receipt itself re-verifies
    assert verify_receipt_file(path)["valid"] is True


def test_format_table_smoke(tmp_path: Path) -> None:
    _write(tmp_path / "a.json", _sealed_receipt())
    rows = audit_receipts_dir(tmp_path)
    text = format_evidence_audit_table(rows)
    assert "a.json" in text and "True" in text


def test_committed_receipts_verify_and_index_fresh() -> None:
    """The committed receipts/ directory must audit clean on this branch:
    every sealed receipt verifies, every digest-named file binds, and
    docs/evidence/index.md is the current regen."""
    rows, receipt = run_evidence_audit(REPO_ROOT / "receipts", check_index=True, root=REPO_ROOT)
    assert receipt["payload"]["findings"] == []
    assert receipt["verdict"] == "pass"
    assert check_evidence_index_fresh(REPO_ROOT) is True
