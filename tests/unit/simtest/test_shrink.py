"""Subset-minimization of a failing fault schedule."""

from __future__ import annotations

from quant_fund.simtest.faults import Fault, FaultSchedule, shrink_schedule


def _fails_when_api_and_gap(schedule: FaultSchedule) -> bool:
    kinds = {fault.kind for fault in schedule.faults}
    return "api_5xx" in kinds and "feed_gap" in kinds


def test_shrink_drops_irrelevant_faults() -> None:
    schedule = FaultSchedule(
        (
            Fault(0, "clock_skew", magnitude=1.0),
            Fault(1, "api_5xx"),
            Fault(1, "feed_reorder"),
            Fault(2, "feed_gap", target="BBB"),
            Fault(3, "api_slow"),
            Fault(4, "partial_fill", magnitude=0.4),
        )
    )
    minimized = shrink_schedule(schedule, _fails_when_api_and_gap)
    assert {fault.kind for fault in minimized.faults} == {"api_5xx", "feed_gap"}
    for fault in minimized.faults:
        without = FaultSchedule(tuple(item for item in minimized.faults if item != fault))
        assert _fails_when_api_and_gap(without) is False


def test_shrink_returns_empty_when_the_empty_schedule_fails() -> None:
    schedule = FaultSchedule((Fault(0, "crash"), Fault(1, "disk_full")))
    assert shrink_schedule(schedule, lambda _schedule: True) == FaultSchedule(())


def test_shrink_rejects_a_passing_schedule() -> None:
    try:
        shrink_schedule(FaultSchedule((Fault(0, "crash"),)), lambda _schedule: False)
    except ValueError as exc:
        assert "failing" in str(exc)
    else:
        raise AssertionError("passing schedule was shrunk")
