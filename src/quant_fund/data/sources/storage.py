"""Persist normalized public-source frames with provenance receipts."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_fund.data.sources.base import SourceError


def _safe_destination(root: str | Path, source: str, filename: str | None) -> Path:
    if not source or source in {".", ".."} or "/" in source or "\\" in source:
        raise SourceError("source must be a non-empty path-safe label")
    base = (Path(root) / "raw" / "sources").resolve()
    relative_name = Path(filename or f"{source}.parquet")
    if relative_name.is_absolute():
        raise SourceError("source output filename must be relative")
    destination = (base / relative_name).resolve()
    try:
        destination.relative_to(base)
    except ValueError as exc:
        raise SourceError("source output filename escapes raw/sources") from exc
    if destination.suffix != ".parquet":
        raise SourceError("source output filename must use the .parquet suffix")
    return destination


def _time_range(frame: pl.DataFrame, column: str) -> dict[str, str] | None:
    if column not in frame.columns or frame.is_empty():
        return None
    values = frame.get_column(column).drop_nulls()
    if values.is_empty():
        return None
    return {"min": str(values.min()), "max": str(values.max())}


def write_source_frame(
    frame: pl.DataFrame,
    root: str | Path,
    source: str,
    *,
    filename: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write a normalized frame and a hash/provenance receipt below ``root``."""
    if "source" in frame.columns and not frame.is_empty():
        labels = frame.get_column("source").cast(pl.String).unique().to_list()
        if labels != [source] and set(labels) != {source}:
            raise SourceError(f"frame source labels do not match {source!r}")
    destination = _safe_destination(root, source, filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    receipt: dict[str, Any] = {
        "schema_version": 2,
        "source": source,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "path": str(destination),
        "sha256": digest,
        "rows": frame.height,
        "columns": sorted(frame.columns),
        "pit_ranges": {
            field: bounds
            for field in ("event_time", "available_time", "ingested_time")
            if (bounds := _time_range(frame, field)) is not None
        },
        "provenance": provenance or {},
    }
    receipt_path = destination.with_suffix(".json")
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return {"data": destination, "receipt": receipt_path}
