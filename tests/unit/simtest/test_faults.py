"""Each fault kind is observable and the session still fails closed or recovers."""

from __future__ import annotations

import pytest

from quant_fund.simtest.faults import FAULT_KINDS, Fault, FaultSchedule, schedule_from_seed
from quant_fund.simtest.invariants import check_invariants
from quant_fund.simtest.session import run_session


@pytest.mark.parametrize("kind", FAULT_KINDS)
def test_fault_kind_finishes_closed_or_recovered(kind: str) -> None:
    magnitude = 0.3 if kind == "partial_fill" else -200_000.0
    conflict = kind == "feed_duplicate"
    schedule = FaultSchedule(
        (Fault(step=0, kind=kind, target="AAA", magnitude=magnitude, conflict=conflict),)
    )
    result = run_session(5, n_days=3, schedule=schedule)
    report = check_invariants(result)
    assert report.ok, report.failures
    if kind in {"api_5xx", "disk_full"}:
        assert "fail_closed" in result.outcomes
        assert result.cash is None
    elif kind == "feed_duplicate":
        assert "fail_closed" in result.outcomes
    elif kind in {"crash", "broker_timeout", "corrupt_cache"}:
        assert "recovered" in result.outcomes
    elif kind == "clock_leap":
        assert result.clock_repeats >= 1
    elif kind == "broker_reject":
        assert result.cash is not None
        assert "sim_broker_reject" in result.reject_reasons


def test_schedule_is_a_pure_function_of_the_seed() -> None:
    assert schedule_from_seed(9, 6, max_faults=4) == schedule_from_seed(9, 6, max_faults=4)
    assert schedule_from_seed(9, 6) != schedule_from_seed(10, 6)


def test_negative_clock_skew_blocks_the_bar() -> None:
    schedule = FaultSchedule(
        (Fault(step=0, kind="clock_skew", target="AAA", magnitude=-3 * 86400.0),)
    )
    result = run_session(1, n_days=2, schedule=schedule)
    assert "fail_closed" in result.outcomes
    assert check_invariants(result).ok
    assert result.cash is None
