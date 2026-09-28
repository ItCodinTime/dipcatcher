"""Adversarial context checks; the proven runner remains closed.

ADVERSARIAL R2 §1-W6 (threading): contextvars do not propagate into
``threading.Thread`` workers, so a rogue vault read from a worker thread
during a proven run used to escape the recorder, the watchdog, and the
decision window (repro: 365 future rows, n_reads=0). The tests below pin
the fix: a lock-guarded thread-visible registry mirrors the active run
(fail-loud cross-thread attach), and ``run_context.proven_thread`` is the
blessed context-propagating worker.
"""

from __future__ import annotations

import logging
import threading
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


def test_worker_thread_rogue_read_trips_and_records(store, caplog):
    """ADVERSARIAL R2 §1-W6 regression: a rogue future read from a BARE
    worker thread inside ``decision_window(t)`` now TRIPS (LeakageError in
    the reading thread, against the run's CURRENT decision window via the
    thread-visible registry) and is RECORDED into the run's recorder —
    pre-fix it escaped with n_reads=0."""
    vault, clock = store
    recorder, watchdog = InMemoryRecorder(), LeakageWatchdog()
    observed = {}
    with (
        caplog.at_level(logging.WARNING, logger="quant_fund.pit.vault"),
        run_context.proven_run(recorder, watchdog),
        run_context.decision_window(clock[0]),
    ):

        def rogue_read():
            try:
                # no hooks, bare thread: the W6 attack
                PitVault(vault.root).asof("silver/bars", clock[-1])
            except BaseException as exc:  # capture what the worker sees
                observed["error"] = exc

        worker = threading.Thread(target=rogue_read)
        worker.start()
        worker.join()  # still inside decision_window(clock[0])
    assert isinstance(observed.get("error"), LeakageError)
    # recorded before the watchdog tripped — the read cannot escape the run
    assert recorder.manifest_summary().n_reads == 1
    assert any("cross-thread" in record.message for record in caplog.records)


def test_worker_thread_read_without_window_fails_closed(store):
    """ADVERSARIAL R2 §1-W6: a bare-thread read while a proven run is active
    but NO decision window is current fails closed in the reading thread
    (registry fallback makes the active run visible cross-thread; the
    mandatory-window VaultError then applies there too)."""
    vault, clock = store
    recorder, watchdog = InMemoryRecorder(), LeakageWatchdog()
    observed = {}
    with run_context.proven_run(recorder, watchdog):

        def rogue_read():
            try:
                PitVault(vault.root).asof("silver/bars", clock[0])
            except BaseException as exc:
                observed["error"] = exc

        worker = threading.Thread(target=rogue_read)
        worker.start()
        worker.join()
    assert isinstance(observed.get("error"), VaultError)
    assert "decision window" in str(observed["error"])
    assert recorder.manifest_summary().n_reads == 0


def test_proven_thread_honest_reads_pass(store, caplog):
    """ADVERSARIAL R2 §1-W6 blessed path: ``run_context.proven_thread``
    propagates the proven-run context (recorder/watchdog/decision window)
    into the worker — honest reads are recorded, watchdog-clean, and need
    no cross-thread fallback."""
    vault, clock = store
    recorder, watchdog = InMemoryRecorder(), LeakageWatchdog()
    observed = {}
    with (
        caplog.at_level(logging.WARNING, logger="quant_fund.pit.vault"),
        run_context.proven_run(recorder, watchdog),
        run_context.decision_window(clock[0]),
    ):

        def honest_read():
            observed["context_is_proven"] = run_context.context_is_proven()
            observed["recorder_is_run"] = run_context.active_recorder() is recorder
            observed["decision_time"] = run_context.current_decision_time()
            observed["rows"] = PitVault(vault.root).asof("silver/bars", clock[0]).rows

        worker = run_context.proven_thread(honest_read)
        worker.start()
        worker.join()
    assert observed["context_is_proven"] is True  # contextvars propagated
    assert observed["recorder_is_run"] is True
    assert observed["decision_time"] == clock[0]  # decision window propagated
    assert observed["rows"] > 0
    assert recorder.manifest_summary().n_reads == 1
    assert watchdog.n_observed == 1
    assert not any("cross-thread" in record.message for record in caplog.records)


def test_sequential_proven_runs_thread_registry_isolation(store):
    """The thread-visible registry is scoped to each proven run: after a run
    exits, a bare-thread stray read attaches to NOTHING, and a second proven
    run records exactly its own reads."""
    vault, clock = store
    first, second = InMemoryRecorder(), InMemoryRecorder()
    with run_context.proven_run(first, LeakageWatchdog()), run_context.decision_window(clock[0]):
        vault.asof("silver/bars", clock[0])
    assert first.manifest_summary().n_reads == 1
    # Between runs: registry restored on exit — a worker-thread stray read
    # must attach to nothing and record nothing.
    assert run_context.active_recorder() is None
    assert run_context.current_decision_time() is None
    observed = {}

    def stray_read():
        observed["rows"] = PitVault(vault.root).asof("silver/bars", clock[0]).rows

    worker = threading.Thread(target=stray_read)
    worker.start()
    worker.join()
    assert observed["rows"] > 0
    assert first.manifest_summary().n_reads == 1  # run 1's recorder untouched
    with run_context.proven_run(second, LeakageWatchdog()), run_context.decision_window(clock[-1]):
        vault.asof("silver/bars", clock[-1])
    assert second.manifest_summary().n_reads == 1  # exactly its own reads
    assert run_context.active_recorder() is None
    assert run_context.current_decision_time() is None
