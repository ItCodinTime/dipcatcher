"""Point-in-time alignment of news records to the repo's daily bar grid.

Convention
----------
* A bar grid is the sorted list of bar timestamps the downstream panel
  actually uses — for ``file_us_wide`` daily bars that is each session's
  ``event_time`` (the close print, 20:00 or 21:00 UTC depending on DST).
* A record is *usable* at bar ``t`` iff its knowledge timestamp
  ``effective_ts = max(publish_ts, ingest_ts)`` is **strictly earlier** than
  ``bar_time(t)`` (``strict=True``, the default). A headline timestamped at
  the exact close print cannot have informed that close — it rolls to the
  next session's bar. ``strict=False`` selects the inclusive boundary
  (``effective_ts <= bar_time``) for panels that treat same-instant data as
  contemporaneous.
* Records whose ``effective_ts`` is after the last grid bar are unassigned
  and counted, never silently dropped.

This is the no-lookahead contract: the signal attached to bar ``t`` uses
only records with ``effective_ts < bar_time(t)`` (or ``<=`` under
``strict=False``), so shifting a headline past a boundary can only ever move
its contribution *later* in time — never earlier.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from typing import Iterable

from quant_fund.data.calendars import session_days
from quant_fund.research.news_signals.schema import NewsRecord, normalize_utc


@dataclass(frozen=True)
class AlignedNews:
    """A record paired with the first bar at which it is usable."""

    record: NewsRecord
    bar_time: datetime


@dataclass(frozen=True)
class BarGrid:
    """Sorted unique bar timestamps (tz-aware UTC) forming the decision grid."""

    bar_times: tuple[datetime, ...]

    def __post_init__(self) -> None:
        times = tuple(normalize_utc(t, field_name="bar_time") for t in self.bar_times)
        if not times:
            raise ValueError("BarGrid requires at least one bar time")
        if tuple(sorted(set(times))) != times:
            raise ValueError("bar_times must be strictly increasing and unique")
        object.__setattr__(self, "bar_times", times)

    @classmethod
    def from_times(cls, times: Iterable[datetime]) -> BarGrid:
        return cls(tuple(times))

    @classmethod
    def from_sessions(
        cls,
        start: date,
        end: date,
        *,
        close: time = time(21, 0),
    ) -> BarGrid:
        """Weekday-session grid (the repo's default trading calendar).

        ``close`` is the bar timestamp wall-clock in UTC — for the
        ``file_us_wide`` daily panel that is 21:00 UTC under DST, 20:00 UTC in
        winter; callers aligning to real bars should build the grid from the
        bars' own ``event_time`` column instead of assuming a fixed hour.
        """
        days = session_days(start, end)
        if not days:
            raise ValueError(f"no sessions between {start} and {end}")
        return cls(tuple(datetime.combine(d, close, tzinfo=UTC) for d in days))

    @classmethod
    def from_frame(cls, frame: object, *, column: str = "event_time") -> BarGrid:
        """Grid from a polars frame's timestamp column (e.g. real bars)."""
        import polars as pl

        if not isinstance(frame, pl.DataFrame):
            raise TypeError("frame must be a polars DataFrame")
        if column not in frame.columns:
            raise ValueError(f"frame lacks {column!r} column")
        values = frame[column].unique().sort().to_list()
        return cls(
            tuple(
                v if isinstance(v, datetime) else datetime.combine(v, time.min, tzinfo=UTC)
                for v in values
            )
        )

    def assign(self, effective_ts: datetime, *, strict: bool = True) -> int | None:
        """Index of the first bar at which ``effective_ts`` is usable.

        ``strict=True`` → first bar with ``bar_time > effective_ts``
        (conservative; headlines at the exact print roll forward).
        ``strict=False`` → ``bar_time >= effective_ts``.
        ``None`` when the record postdates the final bar.
        """
        ts = normalize_utc(effective_ts, field_name="effective_ts")
        idx = bisect_right(self.bar_times, ts) if strict else bisect_left(self.bar_times, ts)
        if idx >= len(self.bar_times):
            return None
        return idx

    def bar_time(self, index: int) -> datetime:
        return self.bar_times[index]


def attach_signal_bars(
    records: Iterable[NewsRecord],
    grid: BarGrid,
    *,
    strict: bool = True,
) -> tuple[list[AlignedNews], int]:
    """Assign each record to its first usable bar.

    Returns ``(aligned, n_dropped)`` where ``n_dropped`` counts records whose
    knowledge timestamp falls after the last grid bar — they exist, are
    counted, and are never silently smuggled into an earlier bar.
    """
    aligned: list[AlignedNews] = []
    dropped = 0
    for rec in records:
        idx = grid.assign(rec.effective_ts, strict=strict)
        if idx is None:
            dropped += 1
            continue
        aligned.append(AlignedNews(record=rec, bar_time=grid.bar_times[idx]))
    return aligned, dropped


def lookahead_violations(aligned: Iterable[AlignedNews], *, strict: bool = True) -> int:
    """Count assignments that break the no-lookahead invariant (should be 0).

    Invariant: ``effective_ts < bar_time`` under ``strict=True``;
    ``effective_ts <= bar_time`` under ``strict=False``.
    """
    bad = 0
    for a in aligned:
        eff = a.record.effective_ts
        if (strict and eff >= a.bar_time) or (not strict and eff > a.bar_time):
            bad += 1
    return bad
