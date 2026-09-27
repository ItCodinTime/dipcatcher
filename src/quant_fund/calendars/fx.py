"""FX trading calendar: Sunday 17:00 ET → Friday 17:00 ET weekly sessions.

The interbank FX week opens Sunday 17:00 America/New_York and closes Friday
17:00 ET, with a weekend gap between. Sessions use the New-York-close
convention: the session labeled ``d`` (a weekday) runs from 17:00 ET on the
prior calendar day to 17:00 ET on ``d``. There are five 24-hour sessions per
week; US holidays do not close the FX market (liquidity thins, trading
continues) so this calendar intentionally has no holiday table.

DST correctness: ET offset shifts (02:00 local on transition Sundays) always
fall inside the weekend gap, so every FX session is exactly 24 hours even
across a DST boundary — the UTC bounds still move correctly between
21:00/22:00 UTC opens.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from quant_fund.calendars.sessions import Session

ET = ZoneInfo("America/New_York")
FX_CLOSE_LOCAL = time(17, 0)


class FxCalendar:
    """Five 24h sessions per week, labeled by the ET close date (Mon–Fri)."""

    name = "FX"
    tz = ET

    def is_session_day(self, d: date) -> bool:
        # Session label = the weekday on which the 17:00 ET close lands.
        return d.weekday() < 5

    def session_on(self, d: date) -> Session | None:
        if not self.is_session_day(d):
            return None
        close_utc = datetime.combine(d, FX_CLOSE_LOCAL, tzinfo=ET).astimezone(ZoneInfo("UTC"))
        return Session(date=d, open_utc=close_utc - timedelta(days=1), close_utc=close_utc)

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
        cur = d + timedelta(days=1)
        while not self.is_session_day(cur):
            cur += timedelta(days=1)
        return cur

    def previous_session_day(self, d: date) -> date:
        cur = d - timedelta(days=1)
        while not self.is_session_day(cur):
            cur -= timedelta(days=1)
        return cur


FX = FxCalendar()
