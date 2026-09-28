"""Seeded fault schedules for the research-to-paper simulation.

A schedule is a pure function of ``(seed, n_steps, max_faults)``. Replay does
not re-roll it; the same object is an input to both the recording run and the
replay. Shrinking drops faults until the failure predicate flips, and the
result is subset-minimal: removing any remaining fault makes the predicate false.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

FAULT_KINDS: tuple[str, ...] = (
    "feed_gap",
    "feed_duplicate",
    "feed_reorder",
    "clock_skew",
    "clock_jump",
    "clock_dst",
    "clock_leap",
    "broker_reject",
    "broker_timeout",
    "partial_fill",
    "crash",
    "disk_full",
    "corrupt_cache",
    "api_slow",
    "api_5xx",
)

_NAMES = ("AAA", "BBB", "CCC")
# Distinct stream from the session price generator.
_SCHEDULE_MIX = 0xD157_0001


@dataclass(frozen=True)
class Fault:
    """One injected fault at a session step.

    ``magnitude`` is seconds for clock faults and a fill fraction in (0, 1)
    for ``partial_fill``. ``conflict`` marks a duplicate bar whose prices
    disagree with the original.
    """

    step: int
    kind: str
    target: str = "AAA"
    magnitude: float = 0.0
    conflict: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "kind": self.kind,
            "target": self.target,
            "magnitude": self.magnitude,
            "conflict": self.conflict,
        }


@dataclass(frozen=True)
class FaultSchedule:
    faults: tuple[Fault, ...]

    def at(self, step: int) -> tuple[Fault, ...]:
        return tuple(fault for fault in self.faults if fault.step == step)

    def kinds(self, step: int) -> set[str]:
        return {fault.kind for fault in self.at(step)}

    def __len__(self) -> int:
        return len(self.faults)


def schedule_from_seed(seed: int, n_steps: int, *, max_faults: int = 3) -> FaultSchedule:
    """Draw a deterministic fault list. ``max_faults`` includes zero."""
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if max_faults < 0:
        raise ValueError("max_faults must be non-negative")
    rng = np.random.Generator(np.random.PCG64(int(seed) ^ _SCHEDULE_MIX))
    n_faults = int(rng.integers(0, max_faults + 1))
    faults: list[Fault] = []
    for _ in range(n_faults):
        kind = FAULT_KINDS[int(rng.integers(0, len(FAULT_KINDS)))]
        step = int(rng.integers(0, n_steps))
        target = _NAMES[int(rng.integers(0, len(_NAMES)))]
        if kind == "partial_fill":
            magnitude = float(rng.uniform(0.2, 0.8))
        else:
            magnitude = float(rng.uniform(-3.0 * 86400.0, 3.0 * 86400.0))
        conflict = bool(rng.random() < 0.5)
        faults.append(
            Fault(step=step, kind=kind, target=target, magnitude=magnitude, conflict=conflict)
        )
    faults.sort(
        key=lambda fault: (fault.step, fault.kind, fault.target, fault.magnitude, fault.conflict)
    )
    return FaultSchedule(tuple(faults))


def shrink_schedule(
    schedule: FaultSchedule,
    fails: Callable[[FaultSchedule], bool],
) -> FaultSchedule:
    """Subset-minimize ``schedule`` while ``fails(schedule)`` stays true.

    ``fails`` receives a :class:`FaultSchedule` and returns whether the
    failure is still present. The empty schedule is returned when it fails
    on its own. The result is subset-minimal: dropping any one remaining
    fault makes ``fails`` false.
    """
    items = list(schedule.faults)
    if not fails(FaultSchedule(tuple(items))):
        raise ValueError("shrink_schedule requires a failing schedule")
    if fails(FaultSchedule(())):
        return FaultSchedule(())

    n_parts = 2
    while len(items) >= 2:
        size = max(1, math.ceil(len(items) / n_parts))
        parts = [items[index : index + size] for index in range(0, len(items), size)]
        reduced: list[Fault] | None = None
        next_parts = n_parts
        for index, part in enumerate(parts):
            if fails(FaultSchedule(tuple(part))):
                reduced = list(part)
                next_parts = 2
                break
            complement = [fault for j, chunk in enumerate(parts) if j != index for fault in chunk]
            if fails(FaultSchedule(tuple(complement))):
                reduced = complement
                next_parts = max(n_parts - 1, 2)
                break
        if reduced is None:
            if n_parts >= len(items):
                break
            n_parts = min(len(items), n_parts * 2)
            continue
        items = reduced
        n_parts = next_parts
    return FaultSchedule(tuple(items))
