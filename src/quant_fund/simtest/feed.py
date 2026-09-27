"""Synthetic exchange tape and a faulted delivery feed.

Bars are labeled ``synthetic``. Generation uses only the session runtime's
uniform stream, so a replay rebuilds the same tape. Delivery can drop a
name, append a duplicate, or reverse the day's row order. Normalization
drops exact duplicates and refuses conflicting ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from quant_fund.simtest.faults import Fault, FaultSchedule
from quant_fund.simtest.runtime import DeterministicRuntime

NAMES: tuple[str, ...] = ("AAA", "BBB", "CCC")
_INITIAL = {"AAA": 100.0, "BBB": 80.0, "CCC": 120.0}


@dataclass(frozen=True)
class Bar:
    security_id: str
    event_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def same_prices(self, other: Bar) -> bool:
        return (
            self.open == other.open
            and self.high == other.high
            and self.low == other.low
            and self.close == other.close
            and self.volume == other.volume
        )

    def as_row(self) -> dict[str, object]:
        adv = self.close * self.volume
        return {
            "security_id": self.security_id,
            "event_time": self.event_time,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "close_total_return": self.close,
            "volume": self.volume,
            "adv": adv,
            "vol_20": 0.02,
            "source": "synthetic",
        }

    def as_trace(self) -> dict[str, object]:
        return {
            "security_id": self.security_id,
            "event_time": self.event_time,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


def business_calendar(start: datetime, n_bars: int) -> list[datetime]:
    """Weekday closes at 21:00 UTC, which crosses the 2026 US DST change."""
    if start.tzinfo is None:
        raise ValueError("calendar start must be timezone-aware")
    cursor = datetime(start.year, start.month, start.day, tzinfo=UTC)
    out: list[datetime] = []
    while len(out) < n_bars:
        if cursor.weekday() < 5:
            out.append(datetime(cursor.year, cursor.month, cursor.day, 21, 0, tzinfo=UTC))
        cursor += timedelta(days=1)
    return out


def generate_tape(runtime: DeterministicRuntime, times: list[datetime]) -> list[list[Bar]]:
    """One bar list per timestamp. Prices are a seeded multiplicative walk."""
    price = dict(_INITIAL)
    days: list[list[Bar]] = []
    for moment in times:
        row: list[Bar] = []
        for name in NAMES:
            open_px = price[name]
            shock = (runtime.uniform() - 0.5) * 0.02
            close_px = max(open_px * (1.0 + shock), 0.5)
            high = max(open_px, close_px) * 1.001
            low = min(open_px, close_px) * 0.999
            price[name] = close_px
            row.append(
                Bar(
                    security_id=name,
                    event_time=moment,
                    open=open_px,
                    high=high,
                    low=low,
                    close=close_px,
                    volume=1_000_000.0,
                )
            )
        days.append(row)
    return days


def apply_delivery_faults(
    rows: list[Bar],
    faults: tuple[Fault, ...],
) -> list[Bar]:
    """Apply feed faults for one delivery. Conflicting duplicates are marked."""
    delivered = list(rows)
    for fault in faults:
        if fault.kind == "feed_gap":
            delivered = [bar for bar in delivered if bar.security_id != fault.target]
        elif fault.kind == "feed_reorder":
            delivered = list(reversed(delivered))
        elif fault.kind == "feed_duplicate" and delivered:
            source = next(
                (bar for bar in delivered if bar.security_id == fault.target), delivered[0]
            )
            if fault.conflict:
                close_px = source.close * 1.01
                dup = Bar(
                    security_id=source.security_id,
                    event_time=source.event_time,
                    open=source.open * 1.01,
                    high=max(source.high, close_px),
                    low=source.low,
                    close=close_px,
                    volume=source.volume,
                )
            else:
                dup = source
            delivered.append(dup)
    return delivered


def normalize_bars(rows: list[Bar]) -> tuple[list[Bar], str]:
    """Sort and exact-dedupe. Conflicting duplicates fail closed.

    Returns ``(bars, status)`` with status ``ok`` or ``conflict``.
    """
    chosen: dict[tuple[str, str], Bar] = {}
    for bar in rows:
        key = (bar.security_id, bar.event_time.isoformat())
        previous = chosen.get(key)
        if previous is None:
            chosen[key] = bar
            continue
        if not previous.same_prices(bar):
            return [], "conflict"
    ordered = [chosen[key] for key in sorted(chosen)]
    return ordered, "ok"


class SimDataFeed:
    """Indexed tape. Step ``k`` delivers the execution bar ``k + 1``."""

    def __init__(self, runtime: DeterministicRuntime, n_days: int, schedule: FaultSchedule) -> None:
        self._runtime = runtime
        self.schedule = schedule
        self.calendar = business_calendar(datetime(2026, 3, 2, tzinfo=UTC), n_days + 1)
        self._days = generate_tape(runtime, self.calendar)

    def opening_bars(self) -> list[Bar]:
        """First close, known before any decision is filled."""
        return self._deliver(0, self._days[0], tag="open")

    def execution_bars(self, step: int) -> list[Bar]:
        """Bar that a decision at ``step`` would execute against (next close's open)."""
        faults = self.schedule.at(step)
        delivered = apply_delivery_faults(self._days[step + 1], faults)
        return self._deliver(step, delivered, tag="exec")

    def _deliver(self, step: int, rows: list[Bar], *, tag: str) -> list[Bar]:
        payload = [bar.as_trace() for bar in rows]
        recorded = self._runtime.effect("feed", {"step": step, "tag": tag}, payload)
        if not isinstance(recorded, list):
            raise RuntimeError("feed trace did not return a list")
        rebuilt: list[Bar] = []
        for item in recorded:
            rebuilt.append(
                Bar(
                    security_id=str(item["security_id"]),
                    event_time=item["event_time"],
                    open=float(item["open"]),
                    high=float(item["high"]),
                    low=float(item["low"]),
                    close=float(item["close"]),
                    volume=float(item["volume"]),
                )
            )
        return rebuilt
