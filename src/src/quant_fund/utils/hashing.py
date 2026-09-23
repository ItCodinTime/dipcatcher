"""Reproducible dataset fingerprints."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonicalize(value: Any) -> Any:
    """Convert common research values to a strict, stable JSON representation."""
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, bytes):
        return value.hex()
    # Numpy/Pandas scalar values expose ``item`` without requiring either
    # package as a dependency of this small utility module.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _canonicalize(item())
        except (TypeError, ValueError):
            pass
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize owned metadata deterministically for lineage fingerprints."""
    return json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def canonical_frame_fingerprint(frame: Any) -> str:
    """Hash a materialized tabular frame independent of row/column ordering.

    The function intentionally uses only the frame's public ``columns``,
    ``schema`` and ``to_dicts`` protocol, so the hashing layer does not depend
    on a particular dataframe implementation. Duplicate rows remain counted.
    """
    columns = sorted(str(column) for column in frame.columns)
    selected = frame.select(columns)
    records = [_canonicalize(row) for row in selected.to_dicts()]
    records.sort(key=canonical_json_bytes)
    schema = {
        str(column): str(selected.schema[column]) for column in columns if column in selected.schema
    }
    return hash_bytes(canonical_json_bytes({"columns": columns, "schema": schema, "rows": records}))


def fingerprint(
    *,
    row_count: int,
    min_timestamp: str,
    max_timestamp: str,
    columns: list[str],
    feature_version: str,
    universe_version: str,
    label_version: str,
    extra: dict[str, Any] | None = None,
) -> str:
    payload = {
        "row_count": row_count,
        "min_timestamp": min_timestamp,
        "max_timestamp": max_timestamp,
        "columns": columns,
        "feature_version": feature_version,
        "universe_version": universe_version,
        "label_version": label_version,
        "extra": extra or {},
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hash_bytes(blob)
