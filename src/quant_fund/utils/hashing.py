"""Reproducible dataset fingerprints."""

from __future__ import annotations

import hashlib
import json
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
