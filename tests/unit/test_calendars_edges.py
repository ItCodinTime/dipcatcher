"""Wave 37: calendars session_days / next|previous session / next_open edges.

Research/infrastructure only — no live broker / vendor MD.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl

from quant_fund.data.calendars import (
    add_next_open_column,
    as_date,
    is_weekend,
    next_session,
    previous_session,
    session_days,
)

RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False


def test_is_weekend_sat_sun() -> None:
    assert is_weekend(date(2024, 1, 6))  # Saturday
    assert is_weekend(date(2024, 1, 7))  # Sunday
    assert not is_weekend(date(2024, 1, 5))  # Friday
    assert not is_weekend(date(2024, 1, 8))  # Monday


def test_session_days_skips_weekend() -> None:
    days = session_days(date(2024, 1, 5), date(2024, 1, 8))
    assert days == [date(2024, 1, 5), date(2024, 1, 8)]


def test_session_days_empty_weekend_only_and_inverted_range() -> None:
    assert session_days(date(2024, 1, 6), date(2024, 1, 7)) == []
    assert session_days(date(2024, 1, 8), date(2024, 1, 5)) == []


def test_next_previous_session_over_weekend() -> None:
    assert next_session(date(2024, 1, 5)) == date(2024, 1, 8)
    assert next_session(date(2024, 1, 4)) == date(2024, 1, 5)
    assert previous_session(date(2024, 1, 8)) == date(2024, 1, 5)
    assert previous_session(date(2024, 1, 5)) == date(2024, 1, 4)


def test_as_date_datetime_and_date() -> None:
    assert as_date(datetime(2020, 1, 2, 15, 30, tzinfo=UTC)) == date(2020, 1, 2)
    assert as_date(date(2020, 1, 2)) == date(2020, 1, 2)


def test_add_next_open_column_shift_and_terminal_null() -> None:
    bars = pl.DataFrame(
        {
            "security_id": ["A", "A", "B"],
            "event_time": [
                datetime(2020, 1, 2, tzinfo=UTC),
                datetime(2020, 1, 3, tzinfo=UTC),
                datetime(2020, 1, 2, tzinfo=UTC),
            ],
            "open": [100.0, 99.0, 20.0],
            "high": [101.0, 100.0, 21.0],
            "low": [99.0, 98.0, 19.0],
            "close": [100.5, 99.5, 20.5],
            "volume": [1e6, 1e6, 5e5],
        }
    )
    out = add_next_open_column(bars).sort(["security_id", "event_time"])
    a = out.filter(pl.col("security_id") == "A")
    assert a["next_open"].to_list() == [99.0, None]
    assert a["next_event_time"][0] == datetime(2020, 1, 3, tzinfo=UTC)
    assert a["next_event_time"][1] is None
    b = out.filter(pl.col("security_id") == "B")
    assert b["next_open"][0] is None
