"""Crypto trading calendar: 24/7/365, UTC-day sessions.

Crypto spot markets never close. Sessions are UTC calendar days
``[00:00, 24:00)`` so daily bars label by the UTC date. There are no
holidays, weekends, half days, or DST effects — the exchange clock is UTC.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from quant_fund.calendars.sessions import Session


class CryptoCalendar:
    """Continuous UTC-day sessions; every calendar day is a session day."""

    name = "CRYPTO"
    tz = ZoneInfo("UTC")

    def is_session_day(self, d: date) -> bool:
        return True

    def session_on(self, d: date) -> Session:
        open_utc = datetime.combine(d, time(0, 0), tzinfo=UTC)
        return Session(date=d, open_utc=open_utc, close_utc=open_utc + timedelta(days=1))

    def sessions(self, start: date, end: date) -> list[Session]:
        out: list[Session] = []
        cur = start
        while cur <= end:
            out.append(self.session_on(cur))
            cur += timedelta(days=1)
        return out

    def next_session_day(self, d: date) -> date:
        return d + timedelta(days=1)

    def previous_session_day(self, d: date) -> date:
        return d - timedelta(days=1)


CRYPTO = CryptoCalendar()
