"""NYSE/Nasdaq trading calendar: regular hours, early closes, holidays.

Regular trading hours are 09:30–16:00 America/New_York; early closes are
13:00 ET. Session bounds are built from exchange-local wall times via
``zoneinfo`` so DST transitions are handled structurally — 09:30 ET maps to
14:30 UTC under EST and 13:30 UTC under EDT automatically.

Nasdaq-listed equities follow the same holiday/early-close schedule as NYSE;
``XNYS``/``NYSE``/``NASDAQ`` are aliases of this calendar.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from quant_fund.calendars.holidays import (
    UNSCHEDULED_CLOSURES,
    us_equity_early_closes,
    us_equity_holidays,
)
from quant_fund.calendars.sessions import Session

ET = ZoneInfo("America/New_York")
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)


def _local_to_utc(d: date, t: time) -> datetime:
    """Exchange-local wall time → aware UTC (zoneinfo handles DST)."""
    return datetime.combine(d, t, tzinfo=ET).astimezone(ZoneInfo("UTC"))


class UsEquitiesCalendar:
    """NYSE/Nasdaq session calendar.

    ``sessions(start, end)`` returns sessions for trading-day labels in the
    closed range ``[start, end]`` (exchange-local dates). Weekends, scheduled
    holidays and unscheduled closures have no session; early-close days get a
    shortened session flagged ``is_half_day``.
    """

    name = "XNYS"
    tz = ET

    def __init__(self) -> None:
        self._holiday_cache: dict[int, set[date]] = {}
        self._early_close_cache: dict[int, set[date]] = {}

    def _holidays(self, year: int) -> set[date]:
        if year not in self._holiday_cache:
            self._holiday_cache[year] = us_equity_holidays(year)
        return self._holiday_cache[year]

    def _early_closes(self, year: int) -> set[date]:
        if year not in self._early_close_cache:
            self._early_close_cache[year] = us_equity_early_closes(year, self._holidays(year))
        return self._early_close_cache[year]

    def is_session_day(self, d: date) -> bool:
        if d.weekday() >= 5:
            return False
        if d in UNSCHEDULED_CLOSURES:
            return False
        return d not in self._holidays(d.year)

    def session_on(self, d: date) -> Session | None:
        if not self.is_session_day(d):
            return None
        half = d in self._early_closes(d.year)
        close_local = EARLY_CLOSE if half else REGULAR_CLOSE
        return Session(
            date=d,
            open_utc=_local_to_utc(d, REGULAR_OPEN),
            close_utc=_local_to_utc(d, close_local),
            is_half_day=half,
        )

    def sessions(self, start: date, end: date) -> list[Session]:
        out: list[Session] = []
        cur = start
        while cur <= end:
            session = self.session_on(cur)
            if session is not None:
                out.append(session)
            cur += timedelta(days=1)
        return out

    def next_session_day(self, d: date) -> date:
        """First trading-day label strictly after ``d``."""
        cur = d + timedelta(days=1)
        while not self.is_session_day(cur):
            cur += timedelta(days=1)
        return cur

    def previous_session_day(self, d: date) -> date:
        """Last trading-day label strictly before ``d``."""
        cur = d - timedelta(days=1)
        while not self.is_session_day(cur):
            cur -= timedelta(days=1)
        return cur


XNYS = UsEquitiesCalendar()
"""Shared instance; the calendar is stateless apart from lookup caches."""
