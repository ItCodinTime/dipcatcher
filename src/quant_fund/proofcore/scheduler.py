"""Causal decision scheduler (WAVE2.md §3) — PROOFCORE layer 1.

Pure stdlib + contracts: NO IO, no polars/numpy, no vault imports. Turns a
frozen ``RunSpec`` into a strict, hashed sequence of decision windows. The
grid is fixed-step arithmetic on the tz-aware start (``start + i*step``);
calendar-aware grids are an explicitly documented follow-up, not this wave.

Determinism contract: identical ``RunSpec`` -> identical windows, always.
Window seeds derive from the top-level seed only:
``int.from_bytes(sha256(f"{spec.seed}|{seq}")[:8], "big")`` — no per-window
randomness is ever stored.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta

from quant_fund.proofcore.contracts import ProofError, RunSpec

__all__ = ["DecisionWindow", "Scheduler", "parse_step", "window_seed"]

_STEP_RE = re.compile(r"^([1-9][0-9]*)([dhms])\Z")
_STEP_UNIT = {
    "d": "days",
    "h": "hours",
    "m": "minutes",
    "s": "seconds",
}


def parse_step(s: str) -> timedelta:
    """Parse a fixed grid step ``Nd`` | ``Nh`` | ``Nm`` | ``Ns`` (integer N >= 1).

    Anything else — zero, negative, fractional, unknown unit, empty — fails
    closed with :class:`ProofError`.
    """
    if not isinstance(s, str):
        raise ProofError(f"decision grid step must be a string, got {type(s).__name__}")
    match = _STEP_RE.fullmatch(s)
    if match is None:
        raise ProofError(
            f"illegal decision grid step {s!r}: expected Nd|Nh|Nm|Ns with integer N >= 1"
        )
    magnitude = int(match.group(1))
    unit = _STEP_UNIT[match.group(2)]
    return timedelta(**{unit: magnitude})


def window_seed(seed: int, seq: int) -> int:
    """Deterministic per-window seed: sha256(f"{seed}|{seq}")[:8] as big-endian int."""
    digest = hashlib.sha256(f"{seed}|{seq}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class DecisionWindow:
    """One causal decision window of a proven run (WAVE2.md §3)."""

    seq: int
    decision_time: datetime
    prior_times: tuple[datetime, ...]  # all earlier decision times, in order
    seed: int  # window_seed(spec.seed, seq)


class Scheduler:
    """Strict, hashed sequence of decision windows for a frozen ``RunSpec``.

    Pure and deterministic: no IO, no clock reads, no randomness. Grid times
    are ``start + i*step`` for ``i in range(count)`` — fixed-step arithmetic
    on the tz-aware start, no calendar awareness claimed.
    """

    def __init__(self, spec: RunSpec) -> None:
        start = spec.decision_grid.start
        if start.tzinfo is None or start.utcoffset() is None:
            raise ProofError("decision grid start must be timezone-aware")
        self._spec = spec
        self._step = parse_step(spec.decision_grid.step)
        self._times: tuple[datetime, ...] = tuple(
            start + i * self._step for i in range(spec.decision_grid.count)
        )

    @property
    def spec(self) -> RunSpec:
        return self._spec

    @property
    def step(self) -> timedelta:
        return self._step

    def __len__(self) -> int:
        return len(self._times)

    def __iter__(self) -> Iterator[DecisionWindow]:
        for seq in range(len(self._times)):
            yield self.window_at(seq)

    def window_at(self, seq: int) -> DecisionWindow:
        """Return window ``seq``; bounds-proof (fail-closed on bad index)."""
        if not isinstance(seq, int) or isinstance(seq, bool):
            raise ProofError(f"window seq must be an int, got {type(seq).__name__}")
        if seq < 0 or seq >= len(self._times):
            raise ProofError(
                f"window seq {seq} out of bounds for grid of {len(self._times)} windows"
            )
        return DecisionWindow(
            seq=seq,
            decision_time=self._times[seq],
            prior_times=self._times[:seq],
            seed=window_seed(self._spec.seed, seq),
        )
