"""NYSE/Nasdaq calendar: holidays, unscheduled closures, early closes, DST.

Research/infrastructure only — no live broker / vendor MD.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from quant_fund.calendars import XNYS, get_calendar
from quant_fund.calendars.holidays import (
    UNSCHEDULED_CLOSURES,
    easter_sunday,
    us_equity_early_closes,
    us_equity_holidays,
)

ET = ZoneInfo("America/New_York")
RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False


def _local(d: date, t: time) -> datetime:
    return datetime.combine(d, t, tzinfo=ET)


# --- scheduled holidays ----------------------------------------------------


@pytest.mark.parametrize(
    "d",
    [
        date(2024, 1, 1),  # New Year's Day
        date(2024, 1, 15),  # MLK Day
        date(2024, 2, 19),  # Washington's Birthday
        date(2024, 3, 29),  # Good Friday
        date(2024, 5, 27),  # Memorial Day
        date(2024, 6, 19),  # Juneteenth
        date(2024, 7, 4),  # Independence Day
        date(2024, 9, 2),  # Labor Day
        date(2024, 11, 28),  # Thanksgiving
        date(2024, 12, 25),  # Christmas
    ],
)
def test_nyse_scheduled_holidays_2024(d: date) -> None:
    assert XNYS.session_on(d) is None
    assert not XNYS.is_session_day(d)


def test_nyse_observed_holidays() -> None:
    # Christmas 2021 fell Saturday -> observed Friday 2021-12-24 (full close).
    assert XNYS.session_on(date(2021, 12, 24)) is None
    # July 4 2020 fell Saturday -> observed Friday 2020-07-03 (not early close).
    assert XNYS.session_on(date(2020, 7, 3)) is None
    # New Year 2023 fell Sunday -> observed Monday 2023-01-02.
    assert XNYS.session_on(date(2023, 1, 2)) is None


def test_new_years_saturday_keeps_dec31_open() -> None:
    # Jan 1 2022 was Saturday; NYSE stays OPEN Friday 2021-12-31 (special rule).
    session = XNYS.session_on(date(2021, 12, 31))
    assert session is not None
    assert not session.is_half_day
    assert session.close_utc == _local(date(2021, 12, 31), time(16, 0)).astimezone(UTC)


def test_weekends_are_not_sessions() -> None:
    assert XNYS.session_on(date(2024, 3, 9)) is None  # Saturday
    assert XNYS.session_on(date(2024, 3, 10)) is None  # Sunday


# --- unscheduled closures ---------------------------------------------------


def test_unscheduled_closures_are_not_sessions() -> None:
    for d in UNSCHEDULED_CLOSURES:
        assert XNYS.session_on(d) is None, d


def test_sandy_closure_2012() -> None:
    assert XNYS.session_on(date(2012, 10, 29)) is None
    assert XNYS.session_on(date(2012, 10, 30)) is None
    # Reopened Wednesday 2012-10-31.
    assert XNYS.session_on(date(2012, 10, 31)) is not None


def test_ford_funeral_2007() -> None:
    assert XNYS.session_on(date(2007, 1, 2)) is None
    assert XNYS.session_on(date(2007, 1, 3)) is not None


def test_september_11_closure_week() -> None:
    for day in (11, 12, 13, 14):
        assert XNYS.session_on(date(2001, 9, day)) is None
    assert XNYS.session_on(date(2001, 9, 10)) is not None
    assert XNYS.session_on(date(2001, 9, 17)) is not None  # reopened Monday


def test_carter_funeral_2025() -> None:
    assert XNYS.session_on(date(2025, 1, 9)) is None


# --- early closes -----------------------------------------------------------


@pytest.mark.parametrize(
    "d",
    [
        date(2024, 7, 3),  # day before Independence Day
        date(2024, 11, 29),  # day after Thanksgiving
        date(2024, 12, 24),  # Christmas Eve
        date(1997, 12, 26),  # ad-hoc: Friday after Thursday Christmas
        date(2003, 12, 26),  # ad-hoc: Friday after Thursday Christmas
    ],
)
def test_early_closes_1300_et(d: date) -> None:
    session = XNYS.session_on(d)
    assert session is not None
    assert session.is_half_day
    assert session.close_utc == _local(d, time(13, 0)).astimezone(UTC)
    assert session.duration == timedelta(hours=3, minutes=30)


def test_july3_not_early_close_when_july4_saturday() -> None:
    # July 4 2015 was Saturday -> July 3 is the observed holiday (full close).
    assert XNYS.session_on(date(2015, 7, 3)) is None
    # July 4 2014 was Friday -> July 3 is a 13:00 early close.
    session = XNYS.session_on(date(2014, 7, 3))
    assert session is not None and session.is_half_day


def test_dec24_not_early_close_when_christmas_saturday() -> None:
    # Dec 25 2021 Saturday -> Dec 24 is the observed holiday, not a half day.
    assert XNYS.session_on(date(2021, 12, 24)) is None
    early = us_equity_early_closes(2021)
    assert date(2021, 12, 24) not in early


def test_early_close_sets_are_within_year() -> None:
    early_2024 = us_equity_early_closes(2024)
    assert all(d.year == 2024 for d in early_2024)
    assert date(2024, 11, 29) in early_2024
    assert date(1997, 12, 26) not in early_2024


# --- regular session shape + DST --------------------------------------------


def test_regular_session_is_390_minutes() -> None:
    session = XNYS.session_on(date(2024, 3, 8))
    assert session is not None
    assert session.duration == timedelta(minutes=390)
    assert session.open_utc == datetime(2024, 3, 8, 14, 30, tzinfo=UTC)  # EST
    assert session.close_utc == datetime(2024, 3, 8, 21, 0, tzinfo=UTC)


def test_dst_spring_forward_utc_shift() -> None:
    # 2024-03-10 02:00 ET springs forward; Friday EST vs Monday EDT.
    fri = XNYS.session_on(date(2024, 3, 8))
    mon = XNYS.session_on(date(2024, 3, 11))
    assert fri is not None and mon is not None
    assert fri.open_utc == datetime(2024, 3, 8, 14, 30, tzinfo=UTC)
    assert mon.open_utc == datetime(2024, 3, 11, 13, 30, tzinfo=UTC)
    assert mon.duration == timedelta(minutes=390)  # still exactly 6.5h
    assert mon.open_utc.astimezone(ET).time() == time(9, 30)
    assert mon.close_utc.astimezone(ET).time() == time(16, 0)


def test_dst_fall_back_utc_shift() -> None:
    # 2024-11-03 02:00 ET falls back; Friday EDT vs Monday EST.
    fri = XNYS.session_on(date(2024, 11, 1))
    mon = XNYS.session_on(date(2024, 11, 4))
    assert fri is not None and mon is not None
    assert fri.open_utc == datetime(2024, 11, 1, 13, 30, tzinfo=UTC)
    assert mon.open_utc == datetime(2024, 11, 4, 14, 30, tzinfo=UTC)
    assert mon.duration == timedelta(minutes=390)


def test_sessions_range_skips_closures_and_weekends() -> None:
    sessions = XNYS.sessions(date(2001, 9, 10), date(2001, 9, 18))
    labels = [s.date for s in sessions]
    assert labels == [
        date(2001, 9, 10),
        date(2001, 9, 17),
        date(2001, 9, 18),
    ]


def test_next_previous_session_day() -> None:
    assert XNYS.next_session_day(date(2024, 12, 24)) == date(2024, 12, 26)
    assert XNYS.next_session_day(date(2012, 10, 26)) == date(2012, 10, 31)
    assert XNYS.previous_session_day(date(2024, 12, 26)) == date(2024, 12, 24)
    # 2007-01-01 holiday + 2007-01-02 Ford funeral + weekend -> Dec 29 2006.
    assert XNYS.previous_session_day(date(2007, 1, 3)) == date(2006, 12, 29)


def test_easter_computus_spot_checks() -> None:
    assert easter_sunday(2024) == date(2024, 3, 31)
    assert easter_sunday(2025) == date(2025, 4, 20)
    assert easter_sunday(2001) == date(2001, 4, 15)


def test_calendar_aliases() -> None:
    assert get_calendar("xnys") is XNYS
    assert get_calendar("NYSE") is XNYS
    assert get_calendar("nasdaq") is XNYS
    with pytest.raises(ValueError, match="unknown calendar"):
        get_calendar("CME")


def test_holidays_do_not_leak_across_years() -> None:
    h2024 = us_equity_holidays(2024)
    h2025 = us_equity_holidays(2025)
    assert all(d.year == 2024 for d in h2024)
    assert all(d.year == 2025 for d in h2025)
