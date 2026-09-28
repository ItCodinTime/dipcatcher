"""check_invariants over hand-built SessionResult states — every failure tag."""

from __future__ import annotations

import pytest

from quant_fund.simtest.invariants import check_invariants
from quant_fund.simtest.session import SessionResult


def _result(**over) -> SessionResult:
    base = dict(
        seed=7,
        n_days=2,
        state_hashes=["h0", "h1"],
        log_bytes=b"log",
        outcomes=["ok", "recovered"],
        clock_repeats=0,
        idempotent_resume=True,
        receipt_errors=[],
        ledger_ok=True,
        conservation_errors=[],
        cash=1.0,
        shares={},
        order_ids=[],
        fill_ids=["f0"],
        fills=[],
        reject_reasons=[],
    )
    base.update(over)
    return SessionResult(**base)


def test_clean_result_passes() -> None:
    out = check_invariants(_result())
    assert out.ok is True
    assert out.failures == ()


def test_research_and_pnl_flags() -> None:
    out = check_invariants(_result(research_only=False, live_pnl_claim=True))
    assert out.ok is False
    assert "research_only" in out.failures
    assert "live_pnl_claim" in out.failures


def test_data_source_must_be_synthetic() -> None:
    out = check_invariants(_result(data_source="file"))
    assert "data_source" in out.failures


@pytest.mark.parametrize("bad", ["crashed", "partial", "denied"])
def test_unknown_outcome_tagged(bad: str) -> None:
    out = check_invariants(_result(outcomes=["ok", bad], state_hashes=["h0", "h1"]))
    assert f"outcome:{bad}" in out.failures


def test_all_known_outcomes_pass() -> None:
    out = check_invariants(
        _result(outcomes=["ok", "fail_closed", "recovered"], state_hashes=["a", "b", "c"])
    )
    assert out.ok is True


def test_non_idempotent_resume() -> None:
    out = check_invariants(_result(idempotent_resume=False))
    assert "resume_not_idempotent" in out.failures


def test_receipt_errors_listed() -> None:
    out = check_invariants(_result(receipt_errors=["e1", "e2"]))
    assert "receipt:e1,e2" in out.failures


def test_ledger_schema_only_when_cash_present() -> None:
    assert "ledger_schema" in check_invariants(_result(ledger_ok=False)).failures
    # No cash row -> ledger schema is not checked.
    assert check_invariants(_result(ledger_ok=False, cash=None)).ok is True


def test_conservation_errors_propagate() -> None:
    out = check_invariants(_result(conservation_errors=["cash_mismatch"]))
    assert "cash_mismatch" in out.failures


def test_duplicate_fill_ids() -> None:
    out = check_invariants(_result(fill_ids=["f", "f"]))
    assert "duplicate_fill_ids" in out.failures


def test_state_hash_count_includes_backtest() -> None:
    # backtest ran -> one extra hash is expected.
    out = check_invariants(_result(outcomes=["ok"], state_hashes=["a", "b"], backtest="done"))
    assert out.ok is True
    # Missing the extra hash -> count violation.
    out2 = check_invariants(_result(outcomes=["ok"], state_hashes=["a"], backtest="done"))
    assert "state_hash_count" in out2.failures


def test_short_trace() -> None:
    out = check_invariants(_result(state_hashes=[], outcomes=[]))
    assert "short_trace" in out.failures
