"""Seeded swarm runner: real small run plus failure/minimize paths."""

from __future__ import annotations

import pytest

from quant_fund.simtest import swarm as swarm_mod
from quant_fund.simtest.faults import schedule_from_seed
from quant_fund.simtest.session import SessionResult
from quant_fund.simtest.swarm import SwarmReport, run_swarm


def test_swarm_rejects_empty_request() -> None:
    with pytest.raises(ValueError, match="positive"):
        run_swarm(0)


def test_swarm_report_ok_property() -> None:
    report = SwarmReport(n_seeds=1, n_days=1, max_faults=1)
    assert report.ok is True


def test_small_real_swarm_aggregates() -> None:
    report = run_swarm(2, n_days=2, base_seed=900, max_faults=2, shrink=False)
    assert report.n_seeds == 2
    assert report.elapsed_seconds > 0.0
    assert report.sessions_per_second > 0.0
    assert report.data_source == "SYNTHETIC"
    assert report.research_only is True
    assert report.live_pnl_claim is False
    if report.ok:
        assert report.log_bytes_max >= report.log_bytes_min > 0
        assert report.outcome_counts
    # Sessions that crashed record failures, never raise out of the swarm.
    for failure in report.failures:
        assert failure.reason
        assert failure.minimized


def _stub_result(outcomes: list[str] | None = None) -> SessionResult:
    done = outcomes or ["ok"]
    return SessionResult(
        seed=0,
        n_days=2,
        state_hashes=[f"h{i}" for i in range(len(done))],
        log_bytes=b"lg",
        outcomes=done,
        clock_repeats=0,
        idempotent_resume=True,
        receipt_errors=[],
        ledger_ok=True,
        conservation_errors=[],
        cash=None,
        shares={},
        order_ids=[],
        fill_ids=[],
        fills=[],
        reject_reasons=[],
    )


def test_exception_seed_records_failure(monkeypatch) -> None:
    calls: list[int] = []

    def boom(seed, *, n_days, schedule, max_faults):
        calls.append(seed)
        raise RuntimeError("session exploded")

    monkeypatch.setattr(swarm_mod, "run_session", boom)
    report = run_swarm(2, n_days=2, base_seed=5, shrink=False)
    assert not report.ok
    assert len(report.failures) == 2
    assert report.failures[0].seed == 5
    assert "RuntimeError" in report.failures[0].reason
    # shrink=False keeps the full schedule as the minimized record.
    assert report.failures[0].minimized == report.failures[0].schedule


def test_invariant_failure_records_and_minimizes(monkeypatch) -> None:
    # Sessions run fine but invariants flag a fake bad outcome.
    monkeypatch.setattr(swarm_mod, "run_session", lambda *a, **k: _stub_result(["bogus"]))
    report = run_swarm(1, n_days=2, base_seed=3, shrink=True)
    assert len(report.failures) == 1
    failure = report.failures[0]
    assert "outcome:bogus" in failure.reason


def test_minimize_falls_back_when_failure_does_not_reproduce(monkeypatch) -> None:
    # Re-running on the same schedule now succeeds -> shrink_schedule has no
    # failing candidate, raises ValueError, and _minimize keeps the schedule.
    schedule = schedule_from_seed(11, 2, max_faults=2)
    monkeypatch.setattr(swarm_mod, "run_session", lambda *a, **k: _stub_result())
    out = swarm_mod._minimize(11, 2, schedule, 2)
    assert out is schedule


def test_outcome_counts_accumulate(monkeypatch) -> None:
    monkeypatch.setattr(
        swarm_mod,
        "run_session",
        lambda *a, **k: _stub_result(["ok", "fail_closed"]),
    )
    report = run_swarm(3, n_days=2, shrink=False)
    assert report.ok
    assert report.outcome_counts == {"ok": 3, "fail_closed": 3}
    assert report.log_bytes_min == 2


def test_exception_in_minimize_marks_candidate_failing(monkeypatch) -> None:
    seen: list[object] = []

    def flaky(seed, *, n_days, schedule, max_faults):
        seen.append(schedule)
        raise RuntimeError("always")

    monkeypatch.setattr(swarm_mod, "run_session", flaky)
    report = run_swarm(1, n_days=2, base_seed=4, shrink=True)
    # The minimizer re-ran the session while shrinking.
    assert len(seen) >= 1
    assert not report.ok
