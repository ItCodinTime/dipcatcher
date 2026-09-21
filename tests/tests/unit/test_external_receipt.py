from __future__ import annotations

import hashlib
import json
from pathlib import Path

from quant_fund.research.external_receipt import verify_receipt


def _write_receipt(root: Path, *, second_value: str = "20.0") -> Path:
    first = root / "VIXCLS_2020-01-01.csv"
    second = root / "VIXCLS_2025-01-02.csv"
    first.write_text(
        "observation_date,VIXCLS_20200101\n2020-01-02,10.0\n2020-01-03,11.0\n",
        encoding="utf-8",
    )
    second.write_text(
        f"observation_date,VIXCLS_20250102\n2020-01-02,10.0\n2020-01-03,{second_value}\n",
        encoding="utf-8",
    )
    snapshots = []
    for path, vintage in ((first, "2020-01-01"), (second, "2025-01-02")):
        snapshots.append(
            {
                "vintage_date": vintage,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "pit_status": "candidate_only",
                "proof_status": "not_proof",
            }
        )
    receipt = root / "receipt.json"
    receipt.write_text(json.dumps({"series": "VIXCLS", "snapshots": snapshots}), encoding="utf-8")
    return receipt


def test_verify_receipt_checks_hashes_and_reports_common_vintage_revisions(tmp_path: Path) -> None:
    receipt = _write_receipt(tmp_path)

    result = verify_receipt(receipt)

    assert result["ok"] is True
    assert result["snapshot_count"] == 2
    assert result["hashes_verified"] == 2
    assert result["revision_pairs"] == [
        {
            "older_vintage": "2020-01-01",
            "newer_vintage": "2025-01-02",
            "common_observations": 2,
            "changed_observations": 1,
            "value_changes": 1,
            "availability_changes": 0,
            "revisions": [
                {
                    "observation_date": "2020-01-03",
                    "older_value": "11.0",
                    "newer_value": "20.0",
                }
            ],
        }
    ]


def test_verify_receipt_rejects_hash_tampering(tmp_path: Path) -> None:
    receipt = _write_receipt(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["snapshots"][0]["sha256"] = "0" * 64
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_receipt(receipt)


def test_verify_receipt_rejects_non_candidate_status(tmp_path: Path) -> None:
    receipt = _write_receipt(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["snapshots"][0]["proof_status"] = "proven"
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="proof_status"):
        verify_receipt(receipt)


def test_verify_receipt_rejects_unexpected_vix_value_column(tmp_path: Path) -> None:
    receipt = _write_receipt(tmp_path)
    snapshot = tmp_path / "VIXCLS_2020-01-01.csv"
    snapshot.write_text(
        "observation_date,WRONG_COLUMN\n2020-01-02,10.0\n2020-01-03,11.0\n",
        encoding="utf-8",
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["snapshots"][0]["sha256"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="value column"):
        verify_receipt(receipt)


def test_verify_receipt_rejects_invalid_observation_date(tmp_path: Path) -> None:
    receipt = _write_receipt(tmp_path)
    snapshot = tmp_path / "VIXCLS_2020-01-01.csv"
    snapshot.write_text(
        "observation_date,VIXCLS_20200101\n2020-02-30,10.0\n2020-01-03,11.0\n",
        encoding="utf-8",
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["snapshots"][0]["sha256"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="invalid observation date"):
        verify_receipt(receipt)
