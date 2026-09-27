"""Wave 34: WallClock edges — tick/now without claiming live fills."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from quant_fund.paper.clock import WallClock

RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False


def test_wall_clock_now_is_utc_aware() -> None:
    clock = WallClock()
    now = clock.now()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)
    assert clock.exhausted() is False


def test_wall_clock_tick_caller_driven_never_none() -> None:
    clock = WallClock()
    t1 = clock.tick()
    t2 = clock.tick()
    assert t1 is not None and t2 is not None
    assert t1.tzinfo is not None
    assert clock.exhausted() is False
    # Wall clock does not exhaust; no live fill claim attached to clock itself.
    assert LIVE_PNL_CLAIM is False


def test_wall_clock_nonzero_step_advances_deterministically() -> None:
    step = timedelta(minutes=5)
    clock = WallClock(step=step)
    t0 = clock.tick()
    assert t0 is not None
    t1 = clock.tick()
    t2 = clock.tick()
    assert t1 == t0 + step
    assert t2 == t0 + 2 * step
    assert clock.exhausted() is False
    # now() remains real wall time (not synthetic step cursor)
    assert isinstance(clock.now(), datetime)
    assert clock.now().tzinfo == UTC
