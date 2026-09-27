"""Pinned, offline XNYS session times for prospective paper-run preflight.

The community-maintained calendar is a deterministic planning input, not an
exchange attestation or evidence that a vendor bar was available at its close.
An independent comparison with published NYSE hours remains necessary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from importlib.metadata import version
from typing import Any
from zoneinfo import ZoneInfo

import exchange_calendars as xcals

CALENDAR_ID = "XNYS"
PACKAGE_VERSION = "4.13.2"
OFFICIAL_HOURS_URL = "https://www.nyse.com/trade/hours-calendars"
_NY_TIME = ZoneInfo("America/New_York")


def _day(value: date | str) -> date:
    if type(value) is date:
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ValueError("session date must be an ISO date")


def _instant(value: datetime | str) -> datetime:
    timestamp = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(timestamp, datetime) or timestamp.tzinfo is None:
        raise ValueError("packet event_time must have an explicit UTC offset")
    if timestamp.utcoffset() is None:
        raise ValueError("packet event_time must have an explicit UTC offset")
    return timestamp.astimezone(UTC)


def _digest(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class XnysSession:
    day: date
    open_utc: datetime
    close_utc: datetime

    def as_dict(self) -> dict[str, str]:
        return {
            "date": self.day.isoformat(),
            "open_utc": self.open_utc.isoformat(),
            "close_utc": self.close_utc.isoformat(),
        }


@dataclass(frozen=True)
class XnysSchedule:
    """Bounded schedule whose exact rows are committed in the freeze receipt."""

    first_date: date
    last_date: date
    sessions: tuple[XnysSession, ...]
    schedule_sha256: str

    def _core(self) -> dict[str, Any]:
        return {
            "kind": "xnys_schedule",
            "schema_version": 1,
            "calendar_id": CALENDAR_ID,
            "package": "exchange_calendars",
            "package_version": PACKAGE_VERSION,
            "first_date": self.first_date.isoformat(),
            "last_date": self.last_date.isoformat(),
            "sessions": [session.as_dict() for session in self.sessions],
            "official_hours_url": OFFICIAL_HOURS_URL,
            "official_comparison_verified": False,
            "forward_evidence_accepted": False,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._core(), "schedule_sha256": self.schedule_sha256}

    def _index(self, value: date | str) -> int:
        day = _day(value)
        for index, session in enumerate(self.sessions):
            if session.day == day:
                return index
        raise ValueError(f"{day.isoformat()} is not a scheduled XNYS session")

    def session(self, value: date | str) -> XnysSession:
        return self.sessions[self._index(value)]

    def validate_close(self, event_time: datetime | str) -> XnysSession:
        """Require the economic close instant, not just self-consistent times."""
        event = _instant(event_time)
        session = self.session(event.astimezone(_NY_TIME).date())
        if event != session.close_utc:
            raise ValueError("close event_time differs from scheduled XNYS close")
        return session

    def validate_open(self, event_time: datetime | str) -> XnysSession:
        event = _instant(event_time)
        session = self.session(event.astimezone(_NY_TIME).date())
        if event != session.open_utc:
            raise ValueError("open event_time differs from scheduled XNYS open")
        return session

    def validate_next_open(
        self, close_event_time: datetime | str, open_event_time: datetime | str
    ) -> XnysSession:
        close = self.validate_close(close_event_time)
        next_index = self._index(close.day) + 1
        if next_index == len(self.sessions):
            raise ValueError("next open exceeds the frozen XNYS schedule")
        opened = self.validate_open(open_event_time)
        if opened != self.sessions[next_index]:
            raise ValueError("open must be the immediately next XNYS session")
        return opened

    def missing_sessions(self, previous: date | str, current: date | str) -> tuple[str, ...]:
        """Scheduled sessions skipped between two observed session dates."""
        start, end = self._index(previous), self._index(current)
        if end <= start:
            raise ValueError("current session must follow previous session")
        return tuple(session.day.isoformat() for session in self.sessions[start + 1 : end])

    def validate_gap(
        self,
        previous: date | str,
        current: date | str,
        claimed_weekday_closures: list[str],
    ) -> None:
        """Reject hidden trading days and caller-declared closures on those days."""
        prior, present = _day(previous), _day(current)
        missing = self.missing_sessions(prior, present)
        if missing:
            raise ValueError(f"missing scheduled XNYS sessions: {', '.join(missing)}")
        scheduled = {session.day for session in self.sessions}
        closed: list[str] = []
        cursor = prior + timedelta(days=1)
        while cursor < present:
            if cursor.weekday() < 5 and cursor not in scheduled:
                closed.append(cursor.isoformat())
            cursor += timedelta(days=1)
        if claimed_weekday_closures != closed:
            raise ValueError("market_closed claims differ from the pinned XNYS calendar")


def materialize_xnys_schedule(first_date: date | str, last_date: date | str) -> XnysSchedule:
    """Build a deterministic UTC schedule; requires explicit bounded dates."""
    first, last = _day(first_date), _day(last_date)
    if first >= last:
        raise ValueError("XNYS schedule requires first_date < last_date")
    installed = version("exchange_calendars")
    if installed != PACKAGE_VERSION:
        raise ValueError(f"exchange_calendars {PACKAGE_VERSION} required, got {installed}")
    calendar = xcals.get_calendar(CALENDAR_ID, start=first.isoformat(), end=last.isoformat())
    sessions: list[XnysSession] = []
    for label, row in calendar.schedule.iterrows():
        sessions.append(
            XnysSession(
                day=label.date(),
                open_utc=row["open"].to_pydatetime().astimezone(UTC),
                close_utc=row["close"].to_pydatetime().astimezone(UTC),
            )
        )
    if len(sessions) < 2 or sessions[0].day < first or sessions[-1].day > last:
        raise ValueError("XNYS schedule bounds contain fewer than two sessions")
    draft = XnysSchedule(first, last, tuple(sessions), "")
    return XnysSchedule(first, last, draft.sessions, _digest(draft._core()))


def verify_xnys_schedule(payload: dict[str, Any]) -> XnysSchedule:
    """Regenerate the full frozen schedule with the pinned package and compare."""
    if not isinstance(payload, dict):
        raise ValueError("XNYS schedule payload must be an object")
    if set(payload) != {
        "kind",
        "schema_version",
        "calendar_id",
        "package",
        "package_version",
        "first_date",
        "last_date",
        "sessions",
        "official_hours_url",
        "official_comparison_verified",
        "forward_evidence_accepted",
        "schedule_sha256",
    }:
        raise ValueError("XNYS schedule payload fields differ from the frozen schema")
    if (
        payload.get("package_version") != PACKAGE_VERSION
        or payload.get("calendar_id") != CALENDAR_ID
        or payload.get("official_comparison_verified") is not False
        or payload.get("forward_evidence_accepted") is not False
    ):
        raise ValueError("XNYS schedule version, exchange, or honesty status changed")
    expected = materialize_xnys_schedule(payload["first_date"], payload["last_date"])
    if payload != expected.as_dict():
        raise ValueError("frozen XNYS schedule differs from pinned package output")
    return expected
