"""Point-in-time helpers."""

from __future__ import annotations

from datetime import datetime
from typing import cast

import polars as pl

from quant_fund.schemas.errors import PointInTimeError
from quant_fund.schemas.pit import assert_pit_safe

PIT_COLS = [
    "event_time",
    "available_time",
    "ingested_time",
    "source",
    "security_id",
    "revision_id",
]


def require_pit_columns(frame: pl.DataFrame) -> None:
    missing = [c for c in PIT_COLS if c not in frame.columns]
    if missing:
        raise PointInTimeError(f"missing PIT columns: {missing}")


def max_available_time(frame: pl.DataFrame) -> datetime:
    require_pit_columns(frame)
    return frame["available_time"].max()  # type: ignore[return-value]


def filter_available(frame: pl.DataFrame, decision_time: datetime) -> pl.DataFrame:
    require_pit_columns(frame)
    return frame.filter(pl.col("available_time") <= decision_time)


def filter_trailing_returns_asof(frame: pl.DataFrame, asof: datetime) -> pl.DataFrame:
    """Drop unpublished restatements from trailing ``ret_1`` covariance inputs.

    GARCH/RGARCH overlays already refuse ``available_time > asof`` (Wave 111).
    Optimizer and portfolio-risk covariance must share that observability
    contract so a late restatement of an earlier return cannot move relative
    risk after the overlay scale is PIT-clean. Frames without
    ``available_time`` keep the legacy event-time path. Null availability
    among otherwise usable ``ret_1`` rows fails closed rather than treating
    an unpublished restatement as observable.
    """
    if "available_time" not in frame.columns:
        return frame
    if frame.is_empty():
        return frame
    if "ret_1" in frame.columns:
        usable = frame.filter(pl.col("ret_1").is_not_null())
        if usable.height and usable["available_time"].null_count() > 0:
            raise PointInTimeError(
                "trailing return covariance has null available_time; "
                "refusing unobservable restatements"
            )
    return frame.filter(pl.col("available_time") <= asof)


def validate_feature_frame(frame: pl.DataFrame, decision_time: datetime) -> None:
    """Reject unavailable feature rows, including mixed as-of snapshots.

    A frame with a ``decision_time`` column is validated row by row because it
    may intentionally contain multiple as-of dates.  Otherwise the supplied
    decision time applies to the whole frame.  Null availability is unsafe:
    the validator cannot establish that the source was observable.
    """
    if frame.is_empty():
        return

    availability_col = (
        "max_source_available_time"
        if "max_source_available_time" in frame.columns
        else "available_time"
        if "available_time" in frame.columns
        else None
    )
    if availability_col is None:
        return

    if "decision_time" in frame.columns:
        invalid = frame.filter(
            pl.col(availability_col).is_null()
            | pl.col("decision_time").is_null()
            | (pl.col(availability_col) > pl.col("decision_time"))
        )
        if invalid.height:
            raise PointInTimeError(
                f"{availability_col} contains null or future values relative to row decision_time"
            )
        return

    invalid = frame.filter(
        pl.col(availability_col).is_null() | (pl.col(availability_col) > pl.lit(decision_time))
    )
    if invalid.height:
        raise PointInTimeError(
            f"{availability_col} contains null or future values relative to {decision_time}"
        )
    max_available = (
        max_available_time(frame)
        if availability_col == "available_time"
        else frame[availability_col].max()
    )
    assert_pit_safe(cast(datetime, max_available), decision_time)
