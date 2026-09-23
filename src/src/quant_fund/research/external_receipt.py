"""Integrity and revision checks for preserved external data receipts.

This module deliberately reports provenance facts only.  It does not promote a
receipt to a benchmark or a SOTA/production claim.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import date as date_type
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


def _snapshot_path(receipt_path: Path, raw_path: str, base_dir: Path | None) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    if base_dir is not None:
        rooted = base_dir / candidate
        if rooted.exists():
            return rooted
    if candidate.exists():
        return candidate
    return receipt_path.parent / candidate


def _observations(
    path: Path, *, expected_value_column: str | None = None
) -> dict[str, Decimal | None]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "observation_date" not in fieldnames:
            raise ValueError(f"{path}: missing observation_date column")
        value_columns = [name for name in fieldnames if name != "observation_date"]
        if len(value_columns) != 1:
            raise ValueError(f"{path}: expected one value column, found {value_columns!r}")
        value_column = value_columns[0]
        if expected_value_column is not None and value_column != expected_value_column:
            raise ValueError(
                f"{path}: unexpected VIXCLS value column {value_column!r}; "
                f"expected {expected_value_column!r}"
            )
        observations: dict[str, Decimal | None] = {}
        for row in reader:
            date = (row.get("observation_date") or "").strip()
            raw_value = (row.get(value_column) or "").strip()
            if not date:
                raise ValueError(f"{path}: blank observation date")
            try:
                date_type.fromisoformat(date)
            except ValueError as exc:
                raise ValueError(f"{path}: invalid observation date {date!r}") from exc
            if date in observations:
                raise ValueError(f"{path}: duplicate observation date {date}")
            if raw_value in {"", "."}:
                observations[date] = None
                continue
            try:
                observations[date] = Decimal(raw_value)
            except InvalidOperation as exc:
                raise ValueError(f"{path}: non-numeric value {raw_value!r}") from exc
    return observations


def verify_receipt(receipt_path: str | Path, *, base_dir: str | Path | None = None) -> dict[str, Any]:
    """Verify frozen snapshot hashes and report pairwise vintage revisions.

    Paths in a receipt are interpreted relative to ``base_dir`` when supplied,
    then the current working directory, and finally the receipt directory.
    ``ok`` means integrity and receipt-schema checks passed; it is not a proof
    status.
    """

    receipt = Path(receipt_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    snapshots = payload.get("snapshots")
    if payload.get("series") != "VIXCLS":
        raise ValueError("receipt series must be VIXCLS")
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError("receipt snapshots must be a non-empty list")

    base = Path(base_dir) if base_dir is not None else None
    entries: list[tuple[str, Path, dict[str, Decimal | None]]] = []
    seen: set[str] = set()
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            raise ValueError("each snapshot must be an object")
        vintage = snapshot.get("vintage_date")
        raw_path = snapshot.get("path")
        expected = snapshot.get("sha256")
        if (
            not isinstance(vintage, str)
            or not vintage
            or not isinstance(raw_path, str)
            or not raw_path
            or not isinstance(expected, str)
            or not expected
        ):
            raise ValueError("snapshot requires non-empty vintage_date, path, and sha256")
        if snapshot.get("pit_status") != "candidate_only":
            raise ValueError("snapshot pit_status must be candidate_only")
        if snapshot.get("proof_status") != "not_proof":
            raise ValueError("snapshot proof_status must be not_proof")
        if vintage in seen:
            raise ValueError(f"duplicate vintage date {vintage}")
        seen.add(vintage)
        path = _snapshot_path(receipt, raw_path, base)
        if not path.is_file():
            raise ValueError(f"snapshot file does not exist: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"sha256 mismatch for {path}: {actual} != {expected}")
        entries.append((vintage, path, _observations(path, expected_value_column=f"VIXCLS_{vintage.replace('-', '')}")))

    entries.sort(key=lambda item: item[0])
    revision_pairs: list[dict[str, Any]] = []
    for (older, _, old_values), (newer, _, new_values) in zip(entries, entries[1:], strict=False):
        common = sorted(set(old_values) & set(new_values))
        revisions = [
            {
                "observation_date": date,
                "older_value": None if old_values[date] is None else str(old_values[date]),
                "newer_value": None if new_values[date] is None else str(new_values[date]),
            }
            for date in common
            if old_values[date] != new_values[date]
        ]
        value_changes = sum(
            old_values[date] is not None and new_values[date] is not None
            for date in common
            if old_values[date] != new_values[date]
        )
        availability_changes = len(revisions) - value_changes
        revision_pairs.append(
            {
                "older_vintage": older,
                "newer_vintage": newer,
                "common_observations": len(common),
                "changed_observations": len(revisions),
                "value_changes": value_changes,
                "availability_changes": availability_changes,
                "revisions": revisions,
            }
        )
    return {
        "ok": True,
        "proof_status": "not_proof",
        "snapshot_count": len(entries),
        "hashes_verified": len(entries),
        "revision_pairs": revision_pairs,
    }
