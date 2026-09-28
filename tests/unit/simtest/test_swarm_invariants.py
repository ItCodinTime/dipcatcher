"""KATs for swarm invariants and run_swarm failure/shrink paths."""

from __future__ import annotations

import pytest

from quant_fund.simtest import swarm as swarm_mod
from quant_fund.simtest.invariants import check_invariants
from quant_fund.simtest.session import SessionResult
from quant_fund.simtest.swarm import run_swarm


def _clean_result(**over: object) -> SessionResult:
    kwargs: dict[str, object] = {
        "seed": 0,
        "n_days": 1,
        "state_hashes": ["h1"],
        "log_bytes": b"x",
        "outcomes": ["ok"],
        "clock_repeats": 0,
        "idempotent_resume": True,
        "receipt_errors": [],
        "ledger_ok": True,
        "conservation_errors": [],
        "cash": None,
        "shares": {},
        "order_ids": [],
        "fill_ids": [],
        "fills": [],
        "reject_reasons": [],
    }
    kwargs.update(over)
    return SessionResult(**kwargs)  # type: ignore[arg-type]


def test_invariants_pass_on_clean_result() -> None:
    report = check_invariants(_clean_result())
    assert report.ok
    assert report.failures == ()


@pytest.mark.parametrize(
    ("over", "expected"),
    [
        ({"research_only": False}, "research_only"),
        ({"live_pnl_claim": True}, "live_pnl_claim"),
        ({"data_source": "vendor"}, "data_source"),
        ({"outcomes": ["crashed"], "state_hashes": ["h1"]}, "outcome:crashed"),
        ({"idempotent_resume": False}, "resume_not_idempotent"),
        ({"receipt_errors": ["bad seal"]}, "receipt:bad seal"),
        ({"cash": 100.0, "ledger_ok": False}, "ledger_schema"),
        ({"conservation_errors": ["cash drift 3.0"]}, "cash drift 3.0"),
        ({"fill_ids": ["f1", "f1"]}, "duplicate_fill_ids"),
        ({"outcomes": ["ok", "ok"], "state_hashes": ["h1"]}, "state_hash_count"),
        ({"state_hashes": []}, "state_hash_count"),
    ],
)
def test_each_invariant_fires(over: dict[str, object], expected: str) -> None:
    report = check_invariants(_clean_result(**over))
    assert not report.ok
    assert any(expected in f for f in report.failures), report.failures


def test_backtest_run_requires_extra_state_hash() -> None:
    result = _clean_result(backtest="ran", state_hashes=["h1", "h2"])
    assert check_invariants(result).ok
    short = _clean_result(backtest="ran", state_hashes=["h1"])
    assert "state_hash_count" in check_invariants(short).failures


def test_run_swarm_rejects_zero_seeds() -> None:
    with pytest.raises(ValueError, match="n_seeds"):
        run_swarm(0)


def test_run_swarm_records_session_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_a: object, **_k: object) -> SessionResult:
        raise RuntimeError("injected session crash")

    monkeypatch.setattr(swarm_mod, "run_session", boom)
    report = run_swarm(2, n_days=2, base_seed=5, shrink=False)
    assert not report.ok
    assert len(report.failures) == 2
    assert all("RuntimeError" in f.reason for f in report.failures)
    # shrink=False: the recorded minimized schedule is the original.
    assert report.failures[0].minimized == report.failures[0].schedule


def test_run_swarm_shrinks_failing_schedules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_a: object, **_k: object) -> SessionResult:
        raise RuntimeError("injected session crash")

    monkeypatch.setattr(swarm_mod, "run_session", boom)
    report = run_swarm(1, n_days=2, base_seed=5, shrink=True)
    assert not report.ok
    # Every subset of the schedule still fails, so the minimizer may reduce
    # it to the empty schedule — and must never grow it.
    assert len(report.failures[0].minimized) <= len(report.failures[0].schedule)


def test_run_swarm_records_invariant_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        swarm_mod,
        "run_session",
        lambda *_a, **_k: _clean_result(outcomes=["crashed"]),
    )
    report = run_swarm(1, n_days=2, base_seed=7, shrink=False)
    assert not report.ok
    assert "outcome:crashed" in report.failures[0].reason


def test_run_swarm_tallies_clean_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        swarm_mod,
        "run_session",
        lambda *_a, **_k: _clean_result(outcomes=["ok", "recovered"], state_hashes=["h1", "h2"]),
    )
    report = run_swarm(3, n_days=2, base_seed=11, shrink=False)
    assert report.ok
    assert report.outcome_counts == {"ok": 3, "recovered": 3}
    assert report.log_bytes_min == report.log_bytes_max == 1
    assert report.research_only is True
    assert report.live_pnl_claim is False
    assert report.data_source == "SYNTHETIC"
