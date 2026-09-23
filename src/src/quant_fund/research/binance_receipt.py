"""Fail-closed verification for preserved Binance candidate receipts."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_fund.data.adapters.binance import BinanceArchiveSpec, normalize_binance_archives


def _resolve(base: Path, value: str) -> Path:
    root = base.resolve()
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("Binance receipt paths must remain within receipt base directory")
    return resolved


def verify_binance_receipt(
    receipt_path: str | Path, *, base_dir: str | Path | None = None
) -> dict[str, Any]:
    """Verify one frozen five-symbol Binance receipt without network access."""
    receipt = Path(receipt_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported Binance receipt schema_version")
    if payload.get("source") != "binance-public-data":
        raise ValueError("receipt source must be binance-public-data")
    for field, expected in (
        ("candidate_only", True),
        ("proof_status", "not_proof"),
        ("proof_eligible", False),
    ):
        if payload.get(field) != expected:
            raise ValueError(f"receipt {field} must be {expected!r}")
    raw_time = payload.get("ingested_time")
    if not isinstance(raw_time, str):
        raise ValueError("receipt ingested_time must be an ISO timestamp")
    try:
        ingested_time = datetime.fromisoformat(raw_time)
    except ValueError as exc:
        raise ValueError("receipt ingested_time must be an ISO timestamp") from exc
    if ingested_time.tzinfo is None:
        raise ValueError("receipt ingested_time must be timezone-aware")
    archives = payload.get("archives")
    if not isinstance(archives, list):
        raise ValueError("receipt archives must be a list")
    root = Path(base_dir) if base_dir is not None else receipt.parent
    specs: list[BinanceArchiveSpec] = []
    for archive in archives:
        if not isinstance(archive, dict):
            raise ValueError("each Binance archive receipt must be an object")
        required = (
            "symbol",
            "date",
            "path",
            "checksum_path",
            "archive_sha256",
            "url",
            "revision_id",
        )
        if any(not isinstance(archive.get(key), str) or not archive[key] for key in required):
            raise ValueError("each Binance archive requires non-empty metadata")
        archive_path = _resolve(root, archive["path"])
        checksum_path = _resolve(root, archive["checksum_path"])
        if not archive_path.is_file() or not checksum_path.is_file():
            raise ValueError("Binance archive and checksum files must exist")
        expected_sha256 = archive["archive_sha256"].lower()
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise ValueError("each Binance archive requires a valid archive_sha256")
        try:
            actual_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError(f"unable to read Binance archive: {archive_path}") from exc
        if actual_sha256 != expected_sha256:
            raise ValueError(f"receipt archive_sha256 mismatch for {archive_path}")
        specs.append(
            BinanceArchiveSpec(
                symbol=archive["symbol"],
                date=archive["date"],
                archive_path=archive_path,
                checksum_path=checksum_path,
                url=archive["url"],
                revision_id=archive["revision_id"],
            )
        )
    frame = normalize_binance_archives(specs, ingested_time=ingested_time.astimezone(UTC))
    return {
        "ok": True,
        "source": "binance-public-data",
        "candidate_only": True,
        "proof_status": "not_proof",
        "proof_eligible": False,
        "archives_verified": len(specs),
        "symbols": frame["symbol"].to_list(),
        "rows": frame.height,
    }
