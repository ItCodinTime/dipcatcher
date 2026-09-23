from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from quant_fund.data.adapters.binance import BINANCE_SYMBOLS
from quant_fund.research.binance_receipt import verify_binance_receipt


def _archive(root: Path, symbol: str, date: str) -> dict[str, str]:
    filename = f"{symbol}-1d-{date}.zip"
    archive = root / filename
    csv_name = filename.removesuffix(".zip") + ".csv"
    row = "1704067200000,100.0,102.0,99.0,101.0,12.5,1704153599999,1250.0,10,6.0,600.0,0\n"
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr(csv_name, row)
    archive.write_bytes(payload.getvalue())
    checksum = root / f"{filename}.CHECKSUM"
    archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum.write_text(
        f"{archive_sha256}  {filename}\n",
        encoding="utf-8",
    )
    return {
        "symbol": symbol,
        "date": date,
        "path": filename,
        "checksum_path": checksum.name,
        "archive_sha256": archive_sha256,
        "url": f"https://data.binance.vision/data/spot/daily/klines/{symbol}/1d/{filename}",
        "revision_id": "current-2024-01-01",
    }

def _receipt(root: Path) -> Path:
    payload = {
        "schema_version": 1,
        "source": "binance-public-data",
        "candidate_only": True,
        "proof_status": "not_proof",
        "proof_eligible": False,
        "ingested_time": "2024-01-02T01:00:00+00:00",
        "archives": [_archive(root, symbol, "2024-01-01") for symbol in BINANCE_SYMBOLS],
    }
    path = root / "receipt.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path

def test_verify_binance_receipt_validates_candidate_archive_set(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)

    result = verify_binance_receipt(receipt, base_dir=tmp_path)

    assert result == {
        "ok": True,
        "source": "binance-public-data",
        "candidate_only": True,
        "proof_status": "not_proof",
        "proof_eligible": False,
        "archives_verified": 5,
        "symbols": sorted(BINANCE_SYMBOLS),
        "rows": 5,
    }


def test_verify_binance_receipt_rejects_stale_archive_provenance(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    archive = tmp_path / payload["archives"][0]["path"]
    archive.write_bytes(archive.read_bytes() + b"preserved-receipt-mutation")
    archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = tmp_path / payload["archives"][0]["checksum_path"]
    checksum.write_text(
        f"{archive_sha256}  {archive.name}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="receipt archive_sha256 mismatch"):
        verify_binance_receipt(receipt, base_dir=tmp_path)


def test_verify_binance_receipt_cli_emits_candidate_result(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/verify_binance_receipt.py",
            str(receipt),
            "--base-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    output = json.loads(result.stdout)
    assert output["ok"] is True
    assert output["candidate_only"] is True
    assert output["proof_status"] == "not_proof"
    assert output["proof_eligible"] is False
    assert result.stderr == ""


def test_verify_binance_receipt_rejects_paths_outside_base_dir(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["archives"][0]["path"] = "../outside.zip"
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must remain within receipt base directory"):
        verify_binance_receipt(receipt, base_dir=tmp_path)


def test_verify_binance_receipt_rejects_malformed_archive_sha256(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["archives"][0]["archive_sha256"] = "not-a-sha"
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="archive_sha256"):
        verify_binance_receipt(receipt, base_dir=tmp_path)


def test_verify_binance_receipt_rejects_missing_archive(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    (tmp_path / payload["archives"][0]["path"]).unlink()

    with pytest.raises(ValueError, match="archive and checksum files must exist"):
        verify_binance_receipt(receipt, base_dir=tmp_path)


def test_crypto_external_benchmark_cli_writes_immutable_candidate_report(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_crypto_external.py",
            "--receipt",
            str(receipt),
            "--base-dir",
            str(tmp_path),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    assert report["source"] == "binance-public-data"
    assert report["candidate_only"] is True
    assert report["proof_status"] == "not_proof"
    assert report["proof_eligible"] is False
    assert report["sota_proven"] is False
    assert report["archives_verified"] == 5
    assert report["protocol"]["symbols"] == sorted(BINANCE_SYMBOLS)
    assert json.loads(output.read_text(encoding="utf-8")) == report

    refused = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_crypto_external.py",
            "--receipt",
            str(receipt),
            "--base-dir",
            str(tmp_path),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "refusing to overwrite" in refused.stderr
