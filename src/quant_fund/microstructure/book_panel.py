"""Vendor-shaped L2 book panels for Northset.

Flat metric panels (not nested levels) are the interchange format. Vendor books
later must emit these columns + PIT timestamps (ADR-021). SYNTHETIC panels from
``synthesize_l2_from_bars`` are the default plumbing substrate.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import polars as pl

from quant_fund.microstructure.book_metrics import (
    DEPTH_SHAPE_FIELDS,
    QUEUE_STRUCTURE_FIELDS,
    SIDE_NOTIONAL_FIELDS,
    SIDE_STRUCTURE_FIELDS,
    TOB_SHARE_FIELDS,
    book_metrics_from_snapshot,
)
from quant_fund.schemas.order_book import OrderBookSnapshot

# Required interchange columns (join keys + top-of-book + depth metrics).
BOOK_PANEL_REQUIRED: tuple[str, ...] = (
    "security_id",
    "event_time",
    "available_time",
    "source",
    "best_bid",
    "best_ask",
    "mid",
    "spread",
    "spread_bps",
    "microprice",
    "microprice_minus_mid",
    "microprice_minus_mid_bps",
    "imbalance_top",
    "imbalance_depth",
    "bid_depth",
    "ask_depth",
    "top_bid_size",
    "top_ask_size",
)

BOOK_PANEL_OPTIONAL: tuple[str, ...] = (
    "revision_id",
    "n_bid_levels",
    "n_ask_levels",
    *DEPTH_SHAPE_FIELDS,
    *SIDE_STRUCTURE_FIELDS,
    *QUEUE_STRUCTURE_FIELDS,
    *SIDE_NOTIONAL_FIELDS,
    *TOB_SHARE_FIELDS,
    "microprice_weight_balance",
    "ofi",
    "queue_imbalance",
    "vpin",
)


def validate_book_panel(frame: pl.DataFrame) -> pl.DataFrame:
    """Fail closed on schema, lineage, uniqueness, values, or PIT violations."""
    missing = [c for c in BOOK_PANEL_REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"book panel missing columns: {missing}")
    if frame.height == 0:
        raise ValueError("book panel is empty")
    for column in ("event_time", "available_time"):
        dtype = frame.schema[column]
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone != "UTC":
            raise ValueError(f"book panel {column} must be timezone-aware UTC Datetime")
    null_required = frame.filter(
        pl.any_horizontal([pl.col(c).is_null() for c in BOOK_PANEL_REQUIRED])
    )
    if null_required.height:
        raise ValueError("book panel has null required values")
    empty_source = frame.filter(pl.col("source").cast(pl.Utf8).str.strip_chars() == "")
    if empty_source.height:
        raise ValueError("book panel has empty source values")
    duplicates = frame.group_by(["security_id", "event_time"]).len().filter(pl.col("len") > 1)
    if duplicates.height:
        raise ValueError("book panel has duplicate (security_id, event_time) snapshots")
    bad_pit = frame.filter(pl.col("available_time") < pl.col("event_time"))
    if bad_pit.height:
        raise ValueError("book panel has available_time < event_time rows")
    numeric = [
        c
        for c in BOOK_PANEL_REQUIRED
        if c not in {"security_id", "event_time", "available_time", "source"}
    ]
    nonfinite = frame.filter(
        pl.any_horizontal([~pl.col(c).cast(pl.Float64).is_finite() for c in numeric])
    )
    if nonfinite.height:
        raise ValueError("book panel has non-finite required metrics")
    positive = (
        "best_bid",
        "best_ask",
        "mid",
        "spread",
        "spread_bps",
        "microprice",
        "bid_depth",
        "ask_depth",
        "top_bid_size",
        "top_ask_size",
    )
    nonpositive = frame.filter(pl.any_horizontal([pl.col(c) <= 0.0 for c in positive]))
    if nonpositive.height:
        raise ValueError("book panel has non-positive price/spread/depth values")
    bad_imbalance = frame.filter(
        (pl.col("imbalance_top").abs() > 1.0) | (pl.col("imbalance_depth").abs() > 1.0)
    )
    if bad_imbalance.height:
        raise ValueError("book panel imbalance must be in [-1, 1]")
    crossed = frame.filter(pl.col("best_bid") >= pl.col("best_ask"))
    if crossed.height:
        raise ValueError("book panel has crossed/locked rows (best_bid >= best_ask)")
    frame = validate_book_panel_depth_honesty(frame)
    return frame.sort(["security_id", "event_time"])


def snapshots_to_panel(snapshots: Iterable[OrderBookSnapshot]) -> pl.DataFrame:
    """Convert validated snapshots into the flat interchange panel."""
    rows: list[dict[str, object]] = []
    for snap in snapshots:
        metrics = book_metrics_from_snapshot(snap)
        rows.append(
            {
                "security_id": snap.security_id,
                "event_time": snap.event_time,
                "available_time": snap.available_time,
                "source": snap.source,
                "revision_id": snap.revision_id,
                **metrics,
            }
        )
    if not rows:
        raise ValueError("no snapshots to convert")
    return validate_book_panel(pl.DataFrame(rows))


def write_book_panel(frame: pl.DataFrame, path: Path | str) -> Path:
    """Validate and write a book panel parquet."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel = validate_book_panel(frame)
    panel.write_parquet(out)
    return out


def load_book_panel(path: Path | str) -> pl.DataFrame:
    """Load and validate a vendor-shaped book panel parquet.

    Missing paths fail closed with ``FileNotFoundError`` (clearer than a raw
    parquet I/O error) so CLI / Northset / providers share one contract.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"order-book panel parquet not found: {target}")
    return validate_book_panel(pl.read_parquet(target))


def validate_book_panel_depth_honesty(frame: pl.DataFrame) -> pl.DataFrame:
    """Fail-closed depth-shape contract when level + shape columns are present.

    Requires ``n_bid_levels`` / ``n_ask_levels``. For each present field in
    ``DEPTH_SHAPE_FIELDS``:

    - ``n_*_levels < 2`` → field must be null/NaN (thin / top-of-book).
    - ``n_*_levels >= 2`` → ``*_log_size_slope`` and ``*_log_price_slope`` must
      be finite. Mean log tick spacings may stay NaN when adjacent levels have
      zero gap (Commander residual #3); thin-side NaN is still required.

    No-op if level columns are absent (legacy top-of-book remaps).
    """
    if "n_bid_levels" not in frame.columns or "n_ask_levels" not in frame.columns:
        return frame
    present = [c for c in DEPTH_SHAPE_FIELDS if c in frame.columns]
    if not present:
        return frame
    if frame.height == 0:
        raise ValueError("book panel is empty")

    def _side(col: str) -> str:
        return "bid" if col.startswith("bid_") else "ask"

    def _n_col(side: str) -> str:
        return f"n_{side}_levels"

    # Thin sides must not carry finite shape metrics.
    for col in present:
        n_col = _n_col(_side(col))
        bad = frame.filter(
            (pl.col(n_col) < 2) & pl.col(col).is_not_null() & pl.col(col).is_not_nan()
        )
        if bad.height:
            raise ValueError(f"depth honesty: {n_col}<2 but {col} finite ({bad.height} rows)")

    # Deep sides: size + price slopes must be finite (tick spacing may be NaN).
    require_finite = [
        c for c in present if c.endswith("_log_size_slope") or c.endswith("_log_price_slope")
    ]
    for col in require_finite:
        n_col = _n_col(_side(col))
        bad = frame.filter((pl.col(n_col) >= 2) & (pl.col(col).is_null() | pl.col(col).is_nan()))
        if bad.height:
            raise ValueError(f"depth honesty: {n_col}>=2 but {col} not finite ({bad.height} rows)")
    return frame
