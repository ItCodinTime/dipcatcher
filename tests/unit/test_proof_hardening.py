"""Adversarial context checks; the proven runner remains closed."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from quant_fund.leakage import LeakageError, LeakageWatchdog
from quant_fund.pit import PitVault, VaultError
from quant_fund.proof.recorder import InMemoryRecorder
from quant_fund.proofcore import run_context
from quant_fund.proofcore.contracts import ProofError
from tests.unit.proof_fake_vault import synthetic_bars


@pytest.fixture()
def store(tmp_path):
    vault = PitVault(tmp_path / "pit")
    vault.create_dataset("silver/bars")
    bars = synthetic_bars(["AAA", "BBB"], 3)
    vault.append("silver/bars", bars)
    return vault, sorted(bars["known_at"].unique().to_list())


def test_future_read_cannot_override_active_decision_time(store):
    vault, clock = store
    recorder, watchdog = InMemoryRecorder(), LeakageWatchdog()
    with (
        run_context.proven_run(recorder, watchdog),
        run_context.decision_window(clock[0]),
        pytest.raises(LeakageError),
    ):
        vault.asof("silver/bars", clock[-1], decision_time=clock[-1])


def test_second_vault_with_private_hooks_cannot_bypass_active_context(store):
    vault, clock = store
    private = InMemoryRecorder()
    rogue = PitVault(vault.root, recorder=private, watchdog=LeakageWatchdog())
    active, watchdog = InMemoryRecorder(), LeakageWatchdog()
    with run_context.proven_run(active, watchdog), run_context.decision_window(clock[0]):
        rogue.asof("silver/bars", clock[0])
    assert active.manifest_summary().n_reads == 1
    assert private.manifest_summary().n_reads == 0
    assert watchdog.n_observed == 1


def test_active_run_without_decision_window_fails_closed(store):
    vault, clock = store
    with (
        run_context.proven_run(InMemoryRecorder(), LeakageWatchdog()),
        pytest.raises(VaultError, match="decision window"),
    ):
        vault.asof("silver/bars", clock[-1], decision_time=clock[-1])


def test_context_hooks_do_not_leak_into_next_run(store):
    vault, clock = store
    first, second = InMemoryRecorder(), InMemoryRecorder()
    with run_context.proven_run(first, LeakageWatchdog()), run_context.decision_window(clock[0]):
        shared = PitVault(vault.root)
        shared.asof("silver/bars", clock[0])
    with run_context.proven_run(second, LeakageWatchdog()), run_context.decision_window(clock[-1]):
        shared.asof("silver/bars", clock[-1])
    assert first.manifest_summary().n_reads == second.manifest_summary().n_reads == 1
    assert run_context.active_recorder() is None
    assert run_context.current_decision_time() is None


def test_nested_context_does_not_inherit_outer_decision_window(store):
    vault, clock = store
    with (
        run_context.proven_run(InMemoryRecorder(), LeakageWatchdog()),
        run_context.decision_window(clock[0]),
    ):
        with (
            run_context.proven_run(InMemoryRecorder(), LeakageWatchdog()),
            pytest.raises(VaultError, match="decision window"),
        ):
            vault.asof("silver/bars", clock[-1])
        assert run_context.current_decision_time() == clock[0]


def test_context_requires_watchdog_and_aware_clock():
    with (
        pytest.raises(ProofError, match="recorder and watchdog"),
        run_context.proven_run(InMemoryRecorder()),
    ):
        pass
    with (
        pytest.raises(ProofError, match="timezone-aware"),
        run_context.decision_window(datetime(2024, 1, 1)),
    ):
        pass
    with run_context.decision_window(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=1)):
        assert run_context.current_decision_time() is not None
