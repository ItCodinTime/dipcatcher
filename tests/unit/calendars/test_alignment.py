"""Bar alignment: grids, slot assignment, DST edges.

Research/infrastructure only — no live broker / vendor MD.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from quant_fund.calendars import (
    CRYPTO,
    FX,
    XNYS,
    assign_bars,
    bars_in_session,
    grid_invariants,
    session_grid,
    slot_for,
)
from quant_fund.calendars.sessions import Session

ET = ZoneInfo("America/New_York")
RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False


def test_full_day_1m_grid_is_390_bars() -> None:
    grid = session_grid(XNYS, date(2024, 3, 8), date(2024, 3, 8), timedelta(minutes=1))
    assert grid.height == 390
    assert grid_invariants(grid) == []
    assert grid["bar_open_utc"].dtype == pl.Datetime("us", "UTC")
    assert grid["bar_open_local"].dtype == pl.Datetime("us", "America/New_York")


def test_half_day_5m_grid_is_42_bars() -> None:
    grid = session_grid(XNYS, date(2024, 7, 3), date(2024, 7, 3), timedelta(minutes=5))
    assert grid.height == 42
    assert grid_invariants(grid) == []
    row = grid.row(-1, named=True)
    assert row["bar_close_utc"] == datetime(2024, 7, 3, 17, 0, tzinfo=UTC)


def test_non_divisible_bar_size_truncates_last_bar() -> None:
    session = XNYS.session_on(date(2024, 3, 8))
    assert session is not None
    bars = bars_in_session(session, timedelta(minutes=7))
    # 390 = 55*7 + 5 -> last bar truncated to the close, never extended past it.
    assert len(bars) == 56
    index, start, end = bars[-1]
    assert index == 55
    assert end == session.close_utc
    assert end - start == timedelta(minutes=5)
    grid = session_grid(XNYS, date(2024, 3, 8), date(2024, 3, 8), timedelta(minutes=7))
    assert grid_invariants(grid) == []


def test_grid_skips_holidays_and_weekends() -> None:
    grid = session_grid(XNYS, date(2024, 12, 24), date(2024, 12, 30), timedelta(hours=1))
    dates = sorted(set(grid["session_date"].to_list()))
    assert dates == [date(2024, 12, 24), date(2024, 12, 26), date(2024, 12, 27), date(2024, 12, 30)]
    assert grid_invariants(grid) == []


def test_grid_empty_range_returns_typed_empty_frame() -> None:
    grid = session_grid(XNYS, date(2024, 3, 9), date(2024, 3, 10), timedelta(minutes=1))
    assert grid.is_empty()
    assert grid_invariants(grid) == []


@pytest.mark.parametrize("size", [timedelta(0), timedelta(microseconds=-1)])
def test_nonpositive_bar_size_fails_even_when_grid_or_input_is_empty(size: timedelta) -> None:
    session = XNYS.session_on(date(2024, 3, 8))
    assert session is not None
    with pytest.raises(ValueError, match="bar_size must be positive"):
        bars_in_session(session, size)
    with pytest.raises(ValueError, match="bar_size must be positive"):
        slot_for(session, session.open_utc, size)
    with pytest.raises(ValueError, match="bar_size must be positive"):
        session_grid(XNYS, date(2024, 3, 9), date(2024, 3, 10), size)
    with pytest.raises(ValueError, match="bar_size must be positive"):
        assign_bars(XNYS, [], size)


def test_local_label_columns_track_et_across_dst() -> None:
    grid = session_grid(XNYS, date(2024, 3, 8), date(2024, 3, 11), timedelta(minutes=30))
    first_per_day = grid.group_by("session_date", maintain_order=True).first()
    opens_et = first_per_day["bar_open_local"].dt.strftime("%H:%M").to_list()
    assert opens_et == ["09:30", "09:30"]  # same wall clock across the transition
    offsets = {
        row["session_date"]: row["bar_open_utc"].hour for row in first_per_day.rows(named=True)
    }
    assert offsets[date(2024, 3, 8)] == 14  # EST: 09:30 ET = 14:30 UTC
    assert offsets[date(2024, 3, 11)] == 13  # EDT: 09:30 ET = 13:30 UTC


def test_slot_for_assigns_and_bounds() -> None:
    session = XNYS.session_on(date(2024, 3, 11))
    assert session is not None
    index, open_, close = slot_for(session, session.open_utc, timedelta(minutes=15))
    assert (index, open_) == (0, session.open_utc)
    index, open_, close = slot_for(
        session, session.close_utc - timedelta(microseconds=1), timedelta(minutes=15)
    )
    assert index == 25  # 390/15 - 1
    assert close == session.close_utc
    with pytest.raises(ValueError, match="outside session"):
        slot_for(session, session.close_utc, timedelta(minutes=15))


def test_assign_bars_maps_and_marks_outside() -> None:
    stamps = [
        datetime(2024, 3, 11, 13, 30, tzinfo=UTC),  # session open -> bar 0
        datetime(2024, 3, 11, 19, 59, tzinfo=UTC),  # last bar
        datetime(2024, 3, 11, 20, 0, tzinfo=UTC),  # close is exclusive -> None
        datetime(2024, 3, 9, 12, 0, tzinfo=UTC),  # Saturday -> None
    ]
    out = assign_bars(XNYS, stamps, timedelta(minutes=30))
    assert out["bar_index"].to_list() == [0, 12, None, None]
    assert out["session_date"].to_list()[:2] == [date(2024, 3, 11)] * 2
    row0 = out.row(0, named=True)
    assert row0["bar_open_utc"] == datetime(2024, 3, 11, 13, 30, tzinfo=UTC)
    assert row0["bar_close_utc"] == datetime(2024, 3, 11, 14, 0, tzinfo=UTC)


def test_assign_bars_refuses_naive() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        assign_bars(XNYS, [datetime(2024, 3, 11, 13, 30)], timedelta(minutes=30))
    with pytest.raises(ValueError, match="positive"):
        assign_bars(XNYS, [datetime(2024, 3, 11, 13, 30, tzinfo=UTC)], timedelta(0))


def test_assign_bars_normalizes_non_utc_input() -> None:
    # 09:30 ET input lands in the same slot as 13:30 UTC.
    ts_et = datetime(2024, 3, 11, 9, 30, tzinfo=ET)
    out = assign_bars(XNYS, [ts_et], timedelta(minutes=30))
    assert out["bar_index"][0] == 0
    assert out["timestamp_utc"][0] == datetime(2024, 3, 11, 13, 30, tzinfo=UTC)


def test_fx_assign_bars_sunday_open_and_friday_close() -> None:
    out = assign_bars(
        FX,
        [
            datetime(2024, 3, 3, 22, 0, tzinfo=UTC),  # Sun 17:00 ET open -> Mon bar 0
            datetime(2024, 3, 8, 22, 0, tzinfo=UTC),  # Fri 17:00 ET close -> None
            datetime(2024, 3, 9, 3, 0, tzinfo=UTC),  # weekend gap -> None
        ],
        timedelta(hours=1),
    )
    assert out["bar_index"].to_list() == [0, None, None]
    assert out["session_date"][0] == date(2024, 3, 4)


def test_crypto_assign_bars() -> None:
    out = assign_bars(CRYPTO, [datetime(2024, 6, 1, 7, 0, tzinfo=UTC)], timedelta(minutes=15))
    assert out["bar_index"][0] == 28


def test_session_requires_utc_bounds() -> None:
    with pytest.raises(ValueError, match="UTC"):
        Session(
            date=date(2024, 1, 2),
            open_utc=datetime(2024, 1, 2, 14, 30, tzinfo=ET),
            close_utc=datetime(2024, 1, 2, 21, 0, tzinfo=UTC),
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        Session(
            date=date(2024, 1, 2),
            open_utc=datetime(2024, 1, 2, 14, 30),
            close_utc=datetime(2024, 1, 2, 21, 0, tzinfo=UTC),
        )
