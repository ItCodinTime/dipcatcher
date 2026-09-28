"""Differential fuzzer: event-driven engine vs ``run_backtest_fast``.

Feeds identical synthetic bar streams to ``_run_backtest_event_loop`` and
``run_backtest_fast`` and asserts fills, reconstructed positions, and NAV
agree within the stated tolerances in ``tests.property._differential``.

Adversarial injections covered by the generator:

* missing prints (dropped mid-panel bars)
* tradinghalts (null opens + kill-switch ``HALT_NEW_ORDERS``)
* zero / negative / NaN prices
* duplicate ``(event_time, security_id)`` bar keys
* splits and cash dividends (raw close vs continuous ``close_total_return``)

When both engines refuse, the exception *types* must match. Known fail-closed
refusals that are intentional product gaps are pinned as explicit xfails in
``tests/regression/test_differential_*`` with a reason — never silent.
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from tests.property._differential import (
    NAV_ATOL,
    NAV_RTOL,
    DiffReport,
    DiffWorkload,
    assemble_workload,
    compare_results,
    run_pair,
)
from tests.property._profiles import adversarial_settings


def _assert_pair(workload: DiffWorkload) -> DiffReport | None:
    """Return a DiffReport on result divergence; None when both refuse alike."""
    ref, fast = run_pair(workload)
    ref_err = isinstance(ref, BaseException)
    fast_err = isinstance(fast, BaseException)
    if ref_err and fast_err:
        assert type(ref) is type(fast), (
            f"exception-type divergence on {workload.notes}: "
            f"ref={type(ref).__name__}:{ref!r} fast={type(fast).__name__}:{fast!r}"
        )
        return None
    if ref_err != fast_err:
        ref_status = f"ERR:{type(ref).__name__}" if ref_err else "ok"
        fast_status = f"ERR:{type(fast).__name__}" if fast_err else "ok"
        ref_exc = repr(ref) if ref_err else None
        fast_exc = repr(fast) if fast_err else None
        pytest.fail(
            f"one-sided failure on {workload.notes}: "
            f"ref={ref_status} fast={fast_status} "
            f"ref_exc={ref_exc} fast_exc={fast_exc}"
        )
    assert not isinstance(ref, BaseException)
    assert not isinstance(fast, BaseException)
    report = compare_results(ref, fast, nav_atol=NAV_ATOL, nav_rtol=NAV_RTOL)
    if not report.ok:
        pytest.fail(f"divergence [{report.reason}] {report.detail} faults={workload.notes}")
    return report


@st.composite
def _workloads(draw: st.DrawFn) -> DiffWorkload:
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    n_assets = draw(st.integers(min_value=1, max_value=4))
    n_days = draw(st.integers(min_value=4, max_value=24))
    # Independently toggle each fault class so Hypothesis can shrink to a
    # single cause when a divergence appears.
    missing = draw(st.booleans())
    halt_days = draw(st.booleans())
    bad_px = draw(st.booleans())
    duplicate = draw(st.booleans())
    split = draw(st.booleans())
    dividend = draw(st.booleans())
    kill = draw(st.booleans())
    # At least one fault class — the clean matched-class path is already
    # covered by test_fast_replay_byte_identity.
    assume(missing or halt_days or bad_px or duplicate or split or dividend or kill)
    return assemble_workload(
        seed=seed,
        n_assets=n_assets,
        n_days=n_days,
        missing_rate=draw(st.floats(0.05, 0.35)) if missing else 0.0,
        n_halt_days=draw(st.integers(1, 3)) if halt_days else 0,
        n_bad_prices=draw(st.integers(1, 4)) if bad_px else 0,
        duplicate=duplicate,
        duplicate_conflict=draw(st.booleans()) if duplicate else False,
        split=split,
        dividend=dividend,
        kill_halt=kill,
    )


@given(workload=_workloads())
@adversarial_settings()
def test_differential_engine_vs_fast_replay(workload: DiffWorkload) -> None:
    """Property: identical streams → identical fills / positions / NAV."""
    _assert_pair(workload)


# --- single-fault smoke pins (shrink targets + coverage documentation) -----


@pytest.mark.parametrize(
    "kwargs",
    [
        {"missing_rate": 0.2},
        {"n_halt_days": 2},
        {"n_bad_prices": 3},
        {"kill_halt": True},
        {"split": True},
        {"dividend": True},
        {"split": True, "dividend": True},
        {"missing_rate": 0.15, "n_halt_days": 1, "kill_halt": True},
        {"n_bad_prices": 2, "split": True},
        {"duplicate": True, "duplicate_conflict": True},
        {"duplicate": True, "duplicate_conflict": False},
    ],
    ids=[
        "missing_prints",
        "halt_days",
        "bad_prices",
        "kill_halt",
        "split",
        "dividend",
        "split_and_dividend",
        "missing_plus_halts",
        "bad_prices_plus_split",
        "duplicate_conflict",
        "duplicate_exact",
    ],
)
def test_differential_single_fault_smoke(kwargs: dict) -> None:
    base = {
        "seed": 42,
        "n_assets": 2,
        "n_days": 12,
        "missing_rate": 0.0,
        "n_halt_days": 0,
        "n_bad_prices": 0,
        "duplicate": False,
        "duplicate_conflict": False,
        "split": False,
        "dividend": False,
        "kill_halt": False,
    }
    base.update(kwargs)
    _assert_pair(assemble_workload(**base))
