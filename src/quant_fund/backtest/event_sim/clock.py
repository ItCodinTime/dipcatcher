"""Deterministic priority-queue clock for the execution simulator.

Ordering key is ``(bar_index, phase, sequence)``. Sequence numbers increase
in schedule order, so two events in the same phase run FIFO. No wall clock
and no unseeded randomness.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class EventKind(IntEnum):
    """Intra-bar phases. Lower values run first."""

    MARKET = 0
    QUEUE = 1
    FILL = 2
    MARK = 3
    SIGNAL = 4
    ORDER = 5
    EXCHANGE = 6


@dataclass(order=True)
class ScheduledEvent:
    bar_index: int
    phase: int
    seq: int
    kind: EventKind = field(compare=False)
    payload: dict[str, Any] = field(compare=False, default_factory=dict)


class EventClock:
    """Min-heap clock. ``seed`` is recorded for run manifests; ordering ignores it."""

    def __init__(self, *, seed: int = 0) -> None:
        if not isinstance(seed, int):
            raise TypeError("seed must be an int")
        self.seed = int(seed)
        self._heap: list[ScheduledEvent] = []
        self._seq = 0

    def schedule(
        self,
        bar_index: int,
        kind: EventKind,
        payload: dict[str, Any] | None = None,
    ) -> ScheduledEvent:
        if bar_index < 0:
            raise ValueError("bar_index must be non-negative")
        event = ScheduledEvent(
            bar_index=int(bar_index),
            phase=int(kind),
            seq=self._seq,
            kind=kind,
            payload={} if payload is None else dict(payload),
        )
        self._seq += 1
        heapq.heappush(self._heap, event)
        return event

    def __len__(self) -> int:
        return len(self._heap)

    def pop(self) -> ScheduledEvent:
        if not self._heap:
            raise IndexError("event clock is empty")
        return heapq.heappop(self._heap)

    def run(self, handler: Callable[[ScheduledEvent], None]) -> list[ScheduledEvent]:
        """Drain the queue. Handlers may schedule further events."""
        seen: list[ScheduledEvent] = []
        while self._heap:
            event = heapq.heappop(self._heap)
            seen.append(event)
            handler(event)
        return seen
