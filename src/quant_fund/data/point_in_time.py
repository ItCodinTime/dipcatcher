"""Point-in-time helpers."""

from __future__ import annotations

from datetime import datetime

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


def validate_feature_frame(frame: pl.DataFrame, decision_time: datetime) -> None:
    if "max_source_available_time" in frame.columns:
        mx = frame["max_source_available_time"].max()
        assert_pit_safe(mx, decision_time)  # type: ignore[arg-type]
        return
    if "available_time" in frame.columns:
        assert_pit_safe(max_available_time(frame), decision_time)
