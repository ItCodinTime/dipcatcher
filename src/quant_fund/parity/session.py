"""Recorded or streamed market-data sessions for parity replay.

A session is an ordered tape of bars. Recorded sessions are reusable.
Streamed sessions are any iterable of :class:`Bar` and are consumed once.
The replay loop never looks past the bar it is currently processing, so a
stream and a recording of the same tape take the same decisions.

This is research infrastructure. It does not connect to a vendor feed and
it does not submit orders.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import polars as pl


def _as_utc(value: object, *, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field} must be a datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _positive(value: object, *, field: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite positive price") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field} must be a finite positive price")
    return number


def _non_negative(value: object, *, field: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite non-negative number") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return number


@dataclass(frozen=True, slots=True)
class Bar:
    """One security's print. ``event_time`` is the bar close.

    ``available_time`` is when the print may be used. A decision at time
    ``t`` can see the bar only when ``available_time <= t``.
    """

    security_id: str
    event_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str
    revision_id: str
    available_time: datetime
    adv: float = 100_000_000.0
    vol_20: float = 0.02

    def __post_init__(self) -> None:
        if not str(self.security_id).strip():
            raise ValueError("security_id must be non-empty")
        object.__setattr__(self, "security_id", str(self.security_id))
        object.__setattr__(self, "event_time", _as_utc(self.event_time, field="event_time"))
        object.__setattr__(
            self, "available_time", _as_utc(self.available_time, field="available_time")
        )
        op = _positive(self.open, field="open")
        hi = _positive(self.high, field="high")
        lo = _positive(self.low, field="low")
        cl = _positive(self.close, field="close")
        if hi < lo:
            raise ValueError("high must be >= low")
        if hi < max(op, cl) or lo > min(op, cl):
            raise ValueError("high/low must contain open and close")
        object.__setattr__(self, "open", op)
        object.__setattr__(self, "high", hi)
        object.__setattr__(self, "low", lo)
        object.__setattr__(self, "close", cl)
        object.__setattr__(self, "volume", _non_negative(self.volume, field="volume"))
        object.__setattr__(self, "adv", _positive(self.adv, field="adv"))
        object.__setattr__(self, "vol_20", _non_negative(self.vol_20, field="vol_20"))
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "revision_id", str(self.revision_id))
        if not self.source:
            raise ValueError("source must be non-empty")


BAR_SCHEMA: dict[str, Any] = {
    "security_id": pl.Utf8,
    "event_time": pl.Datetime(time_zone="UTC"),
    "available_time": pl.Datetime(time_zone="UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "source": pl.Utf8,
    "revision_id": pl.Utf8,
    "adv": pl.Float64,
    "vol_20": pl.Float64,
}


def bars_to_frame(bars: Iterable[Bar]) -> pl.DataFrame:
    """Materialize bars into a stable frame. Empty input keeps the schema."""
    rows = list(bars)
    if not rows:
        return pl.DataFrame(schema=BAR_SCHEMA)
    payload = [
        {
            "security_id": bar.security_id,
            "event_time": bar.event_time,
            "available_time": bar.available_time,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "source": bar.source,
            "revision_id": bar.revision_id,
            "adv": bar.adv,
            "vol_20": bar.vol_20,
        }
        for bar in rows
    ]
    return pl.DataFrame(payload, schema=BAR_SCHEMA)


def iter_bar_groups(bars: Iterable[Bar]) -> Iterator[tuple[Bar, ...]]:
    """Yield ``(event_time)`` groups. Time must be non-decreasing.

    Duplicate ``(event_time, security_id)`` pairs fail closed. Within a
    group, securities are yielded in identifier order.
    """
    buffer: list[Bar] = []
    current: datetime | None = None
    last: datetime | None = None
    for bar in bars:
        if not isinstance(bar, Bar):
            raise TypeError(f"session must yield Bar values, got {type(bar).__name__}")
        if last is not None and bar.event_time < last:
            raise ValueError("session bars must be non-decreasing in event_time")
        if current is None or bar.event_time != current:
            if buffer:
                yield tuple(sorted(buffer, key=lambda item: item.security_id))
            buffer = [bar]
            current = bar.event_time
        else:
            if any(item.security_id == bar.security_id for item in buffer):
                raise ValueError(
                    f"duplicate security_id {bar.security_id!r} at {bar.event_time.isoformat()}"
                )
            buffer.append(bar)
        last = bar.event_time
    if buffer:
        yield tuple(sorted(buffer, key=lambda item: item.security_id))


class MarketSession:
    """Reusable recorded tape. Iterating does not consume it."""

    def __init__(self, bars: Iterable[Bar]) -> None:
        grouped = tuple(iter_bar_groups(bars))
        self.groups: tuple[tuple[Bar, ...], ...] = grouped
        self.bars: tuple[Bar, ...] = tuple(bar for group in grouped for bar in group)
        if not self.bars:
            raise ValueError("market session is empty")

    def __iter__(self) -> Iterator[Bar]:
        return iter(self.bars)

    def __len__(self) -> int:
        return len(self.bars)

    @property
    def synthetic(self) -> bool:
        return any(bar.source.lower() == "synthetic" for bar in self.bars)

    @classmethod
    def from_frame(cls, frame: pl.DataFrame) -> MarketSession:
        """Build a session from a bar frame.

        Required columns: ``security_id, event_time, open, high, low, close,
        volume``. Optional: ``source`` (default ``synthetic``),
        ``revision_id`` (default ``0``), ``available_time`` (default
        ``event_time``), ``adv``, ``vol_20``.
        """
        required = {
            "security_id",
            "event_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"bar frame missing columns: {sorted(missing)}")
        bars: list[Bar] = []
        for row in frame.iter_rows(named=True):
            event_time = _as_utc(row["event_time"], field="event_time")
            available = row["available_time"] if "available_time" in frame.columns else None
            available_time = (
                event_time if available is None else _as_utc(available, field="available_time")
            )
            source = (
                row["source"]
                if "source" in frame.columns and row["source"] is not None
                else "synthetic"
            )
            revision = (
                row["revision_id"]
                if "revision_id" in frame.columns and row["revision_id"] is not None
                else "0"
            )
            adv = row["adv"] if "adv" in frame.columns and row["adv"] is not None else 100_000_000.0
            vol = row["vol_20"] if "vol_20" in frame.columns and row["vol_20"] is not None else 0.02
            bars.append(
                Bar(
                    security_id=str(row["security_id"]),
                    event_time=event_time,
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                    source=str(source),
                    revision_id=str(revision),
                    available_time=available_time,
                    adv=adv,
                    vol_20=vol,
                )
            )
        return cls(bars)

    def to_frame(self) -> pl.DataFrame:
        return bars_to_frame(self.bars)


def end_marks(bars: Iterable[Bar]) -> dict[str, float]:
    """Last close per security, in tape order."""
    marks: dict[str, float] = {}
    for bar in bars:
        marks[bar.security_id] = bar.close
    return marks


def session_fingerprint(bars: Iterable[Bar]) -> str:
    """Stable hash of the tape. Not a research receipt."""

    parts: list[str] = []
    for bar in bars:
        parts.append(
            "|".join(
                (
                    bar.security_id,
                    bar.event_time.isoformat(),
                    bar.available_time.isoformat(),
                    bar.source,
                    bar.revision_id,
                    repr(bar.open),
                    repr(bar.high),
                    repr(bar.low),
                    repr(bar.close),
                    repr(bar.volume),
                )
            )
        )
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()
