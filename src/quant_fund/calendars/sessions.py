"""Core session types for the trading-calendar layer.

A :class:`Session` is one trading session on one exchange-local trading day:
a half-open UTC interval ``[open_utc, close_utc)`` labeled by the
exchange-local trading date. All timestamps are timezone-aware UTC; the
exchange-local wall clock only enters when a calendar *constructs* sessions.

Research/infrastructure only — no live broker or vendor paths.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Protocol, runtime_checkable
from zoneinfo import ZoneInfo


@dataclass(frozen=True, slots=True)
class Session:
    """Half-open session interval, tz-aware UTC internally.

    ``date`` is the trading-day label in exchange-local time. For daily
    markets (NYSE, crypto) it is the local calendar date of the session; for
    FX it is the New-York-close convention (the local date of the 17:00 ET
    close).
    """

    date: date
    open_utc: datetime
    close_utc: datetime
    is_half_day: bool = False

    def __post_init__(self) -> None:
        for name, value in (("open_utc", self.open_utc), ("close_utc", self.close_utc)):
            if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"Session.{name} must be a timezone-aware datetime")
        if self.close_utc <= self.open_utc:
            raise ValueError("Session close must be after open")
        if self.open_utc.utcoffset() != timedelta(0) or self.close_utc.utcoffset() != timedelta(0):
            raise ValueError("Session bounds must be normalized to UTC")

    @property
    def duration(self) -> timedelta:
        return self.close_utc - self.open_utc

    def contains(self, ts: datetime) -> bool:
        """True when aware ``ts`` lies in ``[open_utc, close_utc)``."""
        return self.open_utc <= ts < self.close_utc


@runtime_checkable
class TradingCalendar(Protocol):
    """Calendar protocol: sessions over a closed exchange-local date range."""

    name: str
    tz: ZoneInfo

    def sessions(self, start: date, end: date) -> list[Session]:
        """Sessions whose trading-day label lies in ``[start, end]``."""
        ...

    def is_session_day(self, d: date) -> bool:
        """True when ``d`` is a trading-day label on this calendar."""
        ...

    def session_on(self, d: date) -> Session | None:
        """The session labeled ``d``, or None when ``d`` is not a session day."""
        ...


def require_aware(ts: datetime, *, name: str = "timestamp") -> datetime:
    """Fail closed on naive datetimes; return the instant in UTC."""
    if not isinstance(ts, datetime):
        raise TypeError(f"{name} must be a datetime, got {type(ts).__name__}")
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware (naive datetimes are refused)")
    return ts.astimezone(UTC)


def session_at(sessions: list[Session], ts: datetime) -> Session | None:
    """Locate the session containing aware ``ts`` via binary search.

    ``sessions`` must be sorted by ``open_utc`` and non-overlapping, which
    every calendar in this package guarantees.
    """
    ts = require_aware(ts)
    if not sessions:
        return None
    opens = [s.open_utc for s in sessions]
    idx = bisect_right(opens, ts) - 1
    if idx < 0:
        return None
    candidate = sessions[idx]
    return candidate if candidate.contains(ts) else None


class SessionIndex:
    """Pre-built sorted session table for repeated lookups."""

    def __init__(self, sessions: list[Session]) -> None:
        self._sessions = sorted(sessions, key=lambda s: s.open_utc)
        for prev, nxt in zip(self._sessions[:-1], self._sessions[1:], strict=True):
            if nxt.open_utc < prev.close_utc:
                raise ValueError("SessionIndex requires non-overlapping sessions")
        self._opens = [s.open_utc for s in self._sessions]

    @property
    def sessions(self) -> list[Session]:
        return list(self._sessions)

    def session_at(self, ts: datetime) -> Session | None:
        ts = require_aware(ts)
        idx = bisect_right(self._opens, ts) - 1
        if idx < 0:
            return None
        candidate = self._sessions[idx]
        return candidate if candidate.contains(ts) else None
