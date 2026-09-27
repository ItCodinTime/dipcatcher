"""Simulated clock: skew, jumps, DST, and leap-second repeats.

The exchange timeline (``true_time``) only moves forward. Session time is
``true_time + offset``. Data release uses the exchange timeline, so a clock
that runs fast cannot observe a future bar. A leap fault, or a real
America/New_York fall-back, reports the previous session timestamp again
without advancing the exchange day a second time.

Python ``datetime`` cannot represent a civil ``23:59:60``. The leap fault
is the observable effect of a leap second: the same UTC instant is emitted
twice and tagged ``leap_second=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from quant_fund.simtest.faults import Fault

_NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class ClockReading:
    now: datetime
    true_time: datetime
    repeat: bool
    leap_second: bool
    dst_delta_seconds: int
    real_dst_delta_seconds: int
    offset_seconds: float
    causal_block: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "now": self.now,
            "true_time": self.true_time,
            "repeat": self.repeat,
            "leap_second": self.leap_second,
            "dst_delta_seconds": self.dst_delta_seconds,
            "real_dst_delta_seconds": self.real_dst_delta_seconds,
            "offset_seconds": self.offset_seconds,
            "causal_block": self.causal_block,
        }


class SimClock:
    """Injectable clock. Does not read the wall clock."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("SimClock requires a timezone-aware start")
        self.true_time = start.astimezone(UTC)
        self._offset = timedelta(0)
        # The session has already observed the open. A leap on the first
        # advance therefore repeats this instant instead of being a no-op.
        self._last_now: datetime | None = self.true_time

    def now(self) -> datetime:
        return self.true_time + self._offset

    def advance_to(self, target: datetime, faults: tuple[Fault, ...]) -> ClockReading:
        if target.tzinfo is None:
            raise ValueError("clock target must be timezone-aware")
        target_utc = target.astimezone(UTC)
        if target_utc < self.true_time:
            raise ValueError("exchange time only moves forward")
        previous_offset = _ny_offset_seconds(self.true_time)
        self.true_time = target_utc
        real_delta = _ny_offset_seconds(self.true_time) - previous_offset
        skew = timedelta(0)
        jump = timedelta(0)
        dst = False
        leap = False
        for fault in faults:
            if fault.kind == "clock_skew":
                skew += timedelta(seconds=float(fault.magnitude))
            elif fault.kind == "clock_jump":
                jump += timedelta(seconds=float(fault.magnitude))
            elif fault.kind == "clock_dst":
                dst = True
            elif fault.kind == "clock_leap":
                leap = True
        dst_delta = timedelta(0)
        repeat = False
        if dst:
            if real_delta != 0:
                dst_delta = timedelta(seconds=real_delta)
                if real_delta < 0:
                    repeat = True
            else:
                # No civil transition on this step: still inject a spring-forward
                # hour so the fault is observable on any calendar.
                dst_delta = timedelta(hours=1)
        if leap and self._last_now is not None:
            repeat = True
        self._offset += skew + jump + dst_delta
        session_now = self.now()
        repeated_now = self._last_now if repeat and self._last_now is not None else session_now
        self._last_now = session_now
        return ClockReading(
            now=repeated_now if repeat else session_now,
            true_time=self.true_time,
            repeat=repeat,
            leap_second=leap and repeat,
            dst_delta_seconds=int(dst_delta.total_seconds()),
            real_dst_delta_seconds=real_delta,
            offset_seconds=self._offset.total_seconds(),
            causal_block=session_now < target_utc,
        )


def _ny_offset_seconds(moment: datetime) -> int:
    offset = moment.astimezone(_NY).utcoffset()
    if offset is None:
        return 0
    return int(offset.total_seconds())


def crosses_us_dst(start: datetime, end: datetime) -> int:
    """Return the America/New_York UTC-offset change, in seconds, from start to end."""
    return _ny_offset_seconds(end) - _ny_offset_seconds(start)
