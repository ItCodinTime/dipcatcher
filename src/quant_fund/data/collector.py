"""Explicit, opt-in orchestration for public-source collection.

This boundary is deliberately separate from :mod:`quant_fund.data.ingest`:
collection may use the network, while the normal market ingest pipeline remains
offline and consumes local artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from quant_fund.data.sources import SourceAdapter, SourceError, get_source
from quant_fund.data.sources.storage import write_source_frame

_REQUIRED_PIT_COLUMNS = {
    "event_time",
    "available_time",
    "ingested_time",
    "source",
    "revision_id",
}


@dataclass(frozen=True)
class CollectionResult:
    """Collected frame and durable output paths."""

    frame: pl.DataFrame
    data: Path
    receipt: Path


def _validate_frame(frame: pl.DataFrame, source: str) -> None:
    missing = sorted(_REQUIRED_PIT_COLUMNS.difference(frame.columns))
    if missing:
        raise SourceError(
            f"source {source!r} returned frame missing PIT columns: {', '.join(missing)}"
        )
    if frame.is_empty():
        return
    if frame.get_column("source").cast(pl.String).n_unique() != 1:
        raise SourceError("source frame must contain exactly one source label")
    if frame.get_column("source")[0] != source:
        raise SourceError(f"source frame label does not match requested source {source!r}")
    for field in ("event_time", "available_time", "ingested_time"):
        if frame.get_column(field).null_count() > 0:
            raise SourceError(f"source frame contains null {field}")
    if frame.filter(pl.col("event_time") > pl.col("available_time")).height:
        raise SourceError("source frame violates event_time <= available_time")
    if frame.filter(pl.col("available_time") > pl.col("ingested_time")).height:
        raise SourceError("source frame violates available_time <= ingested_time")


def collect_source(
    source: str,
    root: str | Path,
    *,
    adapter: SourceAdapter | None = None,
    fetch_kwargs: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    filename: str | None = None,
) -> CollectionResult:
    """Fetch one registered source and persist its normalized PIT frame.

    The caller must opt into this function explicitly.  It never changes the
    offline ``make_provider`` routing used by normal ingestion.
    """
    requested = source.strip().lower()
    if not requested:
        raise ValueError("source must be non-empty")
    selected = adapter or get_source(requested)
    canonical = selected.name
    frame = selected.fetch(**(fetch_kwargs or {}))
    if not isinstance(frame, pl.DataFrame):
        raise SourceError(f"source {canonical!r} did not return a Polars DataFrame")
    _validate_frame(frame, canonical)
    paths = write_source_frame(
        frame,
        root,
        canonical,
        filename=filename,
        provenance={"request": fetch_kwargs or {}, **(provenance or {})},
    )
    return CollectionResult(frame=frame, data=paths["data"], receipt=paths["receipt"])
