"""DST-safe bar alignment: session grids and timestamp→slot assignment.

Bars are tiled left-to-right from each session's open. Every bar is
``[bar_open_utc, bar_close_utc)`` with ``bar_open_utc = open + i*bar_size``
and ``bar_close_utc = min(open + (i+1)*bar_size, close)`` — when the session
length is not an exact multiple of ``bar_size`` the final bar is truncated
at the close rather than extended past it (documented policy). All stored
timestamps are tz-aware UTC; exchange-local open/close labels are derived
per row through ``calendar.tz``.

Research/infrastructure only.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl

from quant_fund.calendars.sessions import (
    Session,
    SessionIndex,
    TradingCalendar,
    require_aware,
)


_Schema = dict[str, pl.DataType | type[pl.DataType]]


def grid_schema(tz_name: str = "UTC") -> _Schema:
    """Grid column schema; local label columns are tz-aware in ``tz_name``."""
    return {
        "calendar": pl.String,
        "session_date": pl.Date,
        "is_half_day": pl.Boolean,
        "session_open_utc": pl.Datetime("us", "UTC"),
        "session_close_utc": pl.Datetime("us", "UTC"),
        "bar_index": pl.Int64,
        "bar_open_utc": pl.Datetime("us", "UTC"),
        "bar_close_utc": pl.Datetime("us", "UTC"),
        "bar_open_local": pl.Datetime("us", tz_name),
        "bar_close_local": pl.Datetime("us", tz_name),
    }


ASSIGNMENT_SCHEMA: _Schema = {
    "timestamp_utc": pl.Datetime("us", "UTC"),
    "session_date": pl.Date,
    "bar_index": pl.Int64,
    "bar_open_utc": pl.Datetime("us", "UTC"),
    "bar_close_utc": pl.Datetime("us", "UTC"),
}


def bars_in_session(session: Session, bar_size: timedelta) -> list[tuple[int, datetime, datetime]]:
    """Tile ``[open_utc, close_utc)`` with ``bar_size`` slots.

    The last slot is truncated at ``close_utc`` when the session length is
    not an exact multiple of ``bar_size``; it is never extended beyond the
    close and never produces a zero-length bar.
    """
    if bar_size <= timedelta(0):
        raise ValueError("bar_size must be positive")
    out: list[tuple[int, datetime, datetime]] = []
    start = session.open_utc
    index = 0
    while start < session.close_utc:
        end = min(start + bar_size, session.close_utc)
        out.append((index, start, end))
        index += 1
        start = end
    return out


def session_grid(
    calendar: TradingCalendar,
    start: date,
    end: date,
    bar_size: timedelta,
) -> pl.DataFrame:
    """Full bar grid for every session labeled in ``[start, end]``.

    Columns (see GRID_SCHEMA): UTC bounds for storage plus exchange-local
    ``bar_open_local``/``bar_close_local`` labels. Rows are ordered by
    ``(session_open_utc, bar_index)``.
    """
    sessions = calendar.sessions(start, end)
    schema = grid_schema(calendar.tz.key or "UTC")
    rows: list[dict[str, object]] = []
    for session in sessions:
        for index, bar_open, bar_close in bars_in_session(session, bar_size):
            rows.append(
                {
                    "calendar": calendar.name,
                    "session_date": session.date,
                    "is_half_day": session.is_half_day,
                    "session_open_utc": session.open_utc,
                    "session_close_utc": session.close_utc,
                    "bar_index": index,
                    "bar_open_utc": bar_open,
                    "bar_close_utc": bar_close,
                    "bar_open_local": bar_open.astimezone(calendar.tz),
                    "bar_close_local": bar_close.astimezone(calendar.tz),
                }
            )
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema)


def slot_for(session: Session, ts: datetime, bar_size: timedelta) -> tuple[int, datetime, datetime]:
    """The ``(bar_index, open_utc, close_utc)`` slot containing ``ts``.

    ``ts`` must lie inside the session; the slot index is floor division on
    elapsed UTC time so DST never enters the arithmetic (bounds are already
    absolute instants).
    """
    if not session.contains(ts):
        raise ValueError(f"timestamp {ts!r} is outside session {session.date}")
    index = int((ts - session.open_utc) // bar_size)
    bar_open = session.open_utc + index * bar_size
    bar_close = min(bar_open + bar_size, session.close_utc)
    return index, bar_open, bar_close


def assign_bars(
    calendar: TradingCalendar,
    timestamps: list[datetime],
    bar_size: timedelta,
) -> pl.DataFrame:
    """Assign each aware timestamp to its session bar slot.

    Returns a frame aligned to the input order: ``session_date``,
    ``bar_index`` and bar bounds are null when the timestamp falls outside
    every session (weekends, holidays, the FX weekend gap, pre/post market).
    Naive timestamps are refused — the caller must localize first.
    """
    if bar_size <= timedelta(0):
        raise ValueError("bar_size must be positive")
    stamps = [require_aware(ts, name="timestamp") for ts in timestamps]
    if not stamps:
        return pl.DataFrame(schema=ASSIGNMENT_SCHEMA)
    lo = min(stamps)
    hi = max(stamps)
    # Session labels can differ from the UTC calendar date (FX sessions start
    # on the prior ET day), so pad the label range generously on both sides.
    index = SessionIndex(
        calendar.sessions(lo.date() - timedelta(days=3), hi.date() + timedelta(days=3))
    )
    rows: list[dict[str, object]] = []
    for ts in stamps:
        session = index.session_at(ts)
        if session is None:
            rows.append(
                {
                    "timestamp_utc": ts,
                    "session_date": None,
                    "bar_index": None,
                    "bar_open_utc": None,
                    "bar_close_utc": None,
                }
            )
            continue
        bar_index, bar_open, bar_close = slot_for(session, ts, bar_size)
        rows.append(
            {
                "timestamp_utc": ts,
                "session_date": session.date,
                "bar_index": bar_index,
                "bar_open_utc": bar_open,
                "bar_close_utc": bar_close,
            }
        )
    return pl.DataFrame(rows, schema=ASSIGNMENT_SCHEMA)


def grid_invariants(frame: pl.DataFrame) -> list[str]:
    """Validate a grid frame: returns a list of violation strings (empty = clean).

    Checks the invariants the property tests assert: every bar inside its
    session, tiling without gaps or overlaps, no duplicate slots, monotone
    per-session indexing.
    """
    errors: list[str] = []
    if frame.is_empty():
        return errors
    if frame["bar_open_utc"].dtype != pl.Datetime("us", "UTC") or frame[
        "bar_close_utc"
    ].dtype != pl.Datetime("us", "UTC"):
        errors.append("bar bounds must be tz-aware UTC")
    bad_bounds = frame.filter(
        (pl.col("bar_open_utc") < pl.col("session_open_utc"))
        | (pl.col("bar_close_utc") > pl.col("session_close_utc"))
        | (pl.col("bar_close_utc") <= pl.col("bar_open_utc"))
    )
    if bad_bounds.height:
        errors.append(f"{bad_bounds.height} bars outside/empty within their session")
    ordered = frame.sort("session_open_utc", "bar_index")
    for (_,), session_bars in ordered.group_by(["session_open_utc"], maintain_order=True):
        starts = session_bars["bar_open_utc"].to_list()
        ends = session_bars["bar_close_utc"].to_list()
        indices = session_bars["bar_index"].to_list()
        opens = session_bars["session_open_utc"].to_list()
        closes = session_bars["session_close_utc"].to_list()
        if indices != list(range(len(indices))):
            errors.append(f"non-contiguous bar_index in session {opens[0]!r}")
            continue
        if starts[0] != opens[0]:
            errors.append(f"session {opens[0]!r} first bar does not open at session open")
        if ends[-1] != closes[-1]:
            errors.append(f"session {opens[0]!r} last bar does not end at session close")
        for prev_end, next_start in zip(ends[:-1], starts[1:], strict=True):
            if prev_end != next_start:
                errors.append(f"gap/overlap inside session {opens[0]!r}")
                break
    dupes = frame.group_by("session_open_utc", "bar_index").len().filter(pl.col("len") > 1)
    if dupes.height:
        errors.append(f"{dupes.height} duplicate (session, bar_index) slots")
    return errors
