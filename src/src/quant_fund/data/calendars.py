"""Weekday trading calendar. exchange-calendars is optional later."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl


def is_weekend(d: date) -> bool:
    return d.weekday() >= 5


def session_days(start: date, end: date) -> list[date]:
    out: list[date] = []
    cur = start
    while cur <= end:
        if not is_weekend(cur):
            out.append(cur)
        cur += timedelta(days=1)
    return out


def next_session(d: date) -> date:
    cur = d + timedelta(days=1)
    while is_weekend(cur):
        cur += timedelta(days=1)
    return cur


def previous_session(d: date) -> date:
    cur = d - timedelta(days=1)
    while is_weekend(cur):
        cur -= timedelta(days=1)
    return cur


def as_date(ts: datetime | date) -> date:
    if isinstance(ts, datetime):
        return ts.date()
    return ts


def add_next_open_column(bars: pl.DataFrame) -> pl.DataFrame:
    """For each security, shift open to the next bar (executable after today's close)."""
    return bars.sort(["security_id", "event_time"]).with_columns(
        pl.col("open").shift(-1).over("security_id").alias("next_open"),
        pl.col("event_time").shift(-1).over("security_id").alias("next_event_time"),
    )
