"""Paper clocks: wall-clock or accelerated replay over a bar calendar."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol


class PaperClock(Protocol):
    def now(self) -> datetime: ...

    def tick(self) -> datetime | None:
        """Advance one step; return new time or None when exhausted."""
        ...

    def exhausted(self) -> bool: ...


@dataclass
class WallClock:
    """Wall clock for paper timing. Does not imply live broker fills.

    When ``step`` is zero, ``tick`` records ``now()`` (caller-driven).
    Non-zero ``step`` advances from the previous tick (or first wall ``now()``).
    ``exhausted`` is always False — wall time does not end.
    """

    step: timedelta = field(default_factory=lambda: timedelta(0))
    _last: datetime | None = None

    def now(self) -> datetime:
        return datetime.now(UTC)

    def tick(self) -> datetime | None:
        if self.step:
            base = self._last if self._last is not None else self.now()
            self._last = base + self.step
        else:
            self._last = self.now()
        return self._last

    def exhausted(self) -> bool:
        return False


@dataclass
class ReplayClock:
    """Accelerated replay over discrete event times (e.g. bar closes)."""

    event_times: list[datetime]
    index: int = 0

    def now(self) -> datetime:
        if not self.event_times:
            raise RuntimeError("ReplayClock has no event times")
        i = min(self.index, len(self.event_times) - 1)
        return self.event_times[i]

    def tick(self) -> datetime | None:
        if self.index >= len(self.event_times):
            return None
        t = self.event_times[self.index]
        self.index += 1
        return t

    def exhausted(self) -> bool:
        return self.index >= len(self.event_times)

    def remaining(self) -> int:
        return max(0, len(self.event_times) - self.index)

    def __iter__(self) -> Iterator[datetime]:
        while True:
            t = self.tick()
            if t is None:
                break
            yield t
