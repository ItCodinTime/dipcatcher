"""DST and leap-second behaviour of the simulated clock."""

from __future__ import annotations

from datetime import UTC, datetime

from quant_fund.simtest.clock import SimClock, crosses_us_dst
from quant_fund.simtest.faults import Fault


def test_spring_forward_and_fall_back_match_america_new_york() -> None:
    spring = crosses_us_dst(
        datetime(2026, 3, 6, 21, 0, tzinfo=UTC),
        datetime(2026, 3, 9, 21, 0, tzinfo=UTC),
    )
    fall = crosses_us_dst(
        datetime(2026, 10, 30, 20, 0, tzinfo=UTC),
        datetime(2026, 11, 2, 21, 0, tzinfo=UTC),
    )
    assert spring == 3600
    assert fall == -3600


def test_leap_fault_repeats_the_previous_instant_once() -> None:
    clock = SimClock(datetime(2026, 3, 2, 21, 0, tzinfo=UTC))
    first = clock.advance_to(datetime(2026, 3, 3, 21, 0, tzinfo=UTC), ())
    assert first.repeat is False
    leap = Fault(step=1, kind="clock_leap", magnitude=0.0)
    second = clock.advance_to(datetime(2026, 3, 4, 21, 0, tzinfo=UTC), (leap,))
    assert second.repeat is True
    assert second.leap_second is True
    assert second.now == first.now
    assert second.true_time == datetime(2026, 3, 4, 21, 0, tzinfo=UTC)


def test_dst_fault_on_the_real_spring_transition_jumps_an_hour() -> None:
    clock = SimClock(datetime(2026, 3, 6, 21, 0, tzinfo=UTC))
    reading = clock.advance_to(
        datetime(2026, 3, 9, 21, 0, tzinfo=UTC),
        (Fault(step=0, kind="clock_dst"),),
    )
    assert reading.real_dst_delta_seconds == 3600
    assert reading.dst_delta_seconds == 3600
    assert reading.repeat is False


def test_dst_fault_on_the_real_fall_back_repeats() -> None:
    clock = SimClock(datetime(2026, 10, 30, 20, 0, tzinfo=UTC))
    reading = clock.advance_to(
        datetime(2026, 11, 2, 21, 0, tzinfo=UTC),
        (Fault(step=0, kind="clock_dst"),),
    )
    assert reading.real_dst_delta_seconds == -3600
    assert reading.repeat is True
