"""ADVERSARIAL-driven hardening regressions (proofcore/hardening wave).

- §1b-W1: a strategy reading ``asof(t + 1d)`` while deciding at ``t`` TRIPS
  the watchdog inside a proven run (decision clock from the data, not the
  read's own asof argument).
- §1b-W2: a rogue second ``PitVault(root)`` opened mid-run auto-attaches to
  the active recorder; its reads land in the bundle's data manifest.
- §2-R:   a pristine SIGNED bundle passes ``verify --replay``; tampering
  still fails the HMAC/self-hash checks.

The stub engine (monkeypatched ``_engine_fn``) keeps these tests engine-free;
the stub deterministically derives fills from the weights frame so any input
change flips the proof.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from quant_fund.config.models import AppConfig
from quant_fund.leakage import LeakageError, LeakageWatchdog
from quant_fund.pit import PitVault
from quant_fund.proof.recorder import InMemoryRecorder
from quant_fund.proof.runner import (
    ASOF_SENTINEL,
    BARS_DATASET,
    WEIGHTS_DATASET,
    run_backtest_proven,
)
from quant_fund.proof.sign import HmacSha256Signer
from quant_fund.proofcore import run_context
from tests.unit.proof_fake_vault import synthetic_bars, synthetic_weights

SIDS = ["AAA", "BBB", "CCC"]
N_DAYS = 30


def _build_vault(pit_root: Path, recorder: InMemoryRecorder) -> PitVault:
    vault = PitVault(pit_root, recorder=recorder, watchdog=LeakageWatchdog())
    vault.create_dataset(BARS_DATASET)
    vault.append(BARS_DATASET, synthetic_bars(SIDS, N_DAYS))
    vault.create_dataset(WEIGHTS_DATASET)
    vault.append(WEIGHTS_DATASET, synthetic_weights(SIDS, N_DAYS))
    return vault


def _stub_engine(bars: pl.DataFrame, weights: pl.DataFrame, _config: Any) -> Any:
    """Deterministic minimal engine: two fills whose nav marks derive from
    the weights frame bytes (any input difference changes the proof)."""
    total = float(weights["target_weight"].abs().sum()) if weights.height else 0.0
    t0 = datetime(2024, 1, 2, tzinfo=UTC)
    nav0 = 1_000_000.0 + total
    days = [t0 + timedelta(days=i) for i in range(3)]
    fills = pl.DataFrame(
        {
            "fill_time": days,
            "signal_time": [d - timedelta(days=1) for d in days],
            "security_id": ["AAA", "BBB", "AAA"],
            "quantity": [1.0, -1.0, 2.0],
            "price": [100.0, 101.0, 99.5],
            "fee": [0.1, 0.1, 0.1],
            "spread_cost": [0.05, 0.05, 0.05],
            "impact_cost": [0.02, 0.02, 0.02],
            "decision_price": [99.0, 100.0, 99.0],
        }
    ).with_columns(
        pl.col("fill_time").cast(pl.Datetime("us", "UTC")),
        pl.col("signal_time").cast(pl.Datetime("us", "UTC")),
    )
    equity = pl.DataFrame(
        {
            "event_time": days,
            "nav": [nav0, nav0 * 1.001, nav0 * 1.002],
        }
    ).with_columns(pl.col("event_time").cast(pl.Datetime("us", "UTC")))
    return SimpleNamespace(fills=fills, equity=equity, metrics={"total_return": 0.002})


@pytest.fixture()
def stub_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "quant_fund.proof.runner._engine_fn", lambda _engine: _stub_engine
    )


class _LeakyStrategyVault:
    """Strategy-side vault wrapper: on the FIRST bars decision read at
    decision time t it also reads ``asof(t + 1 day)`` — a genuine future
    read through the real, watchdog-wired vault."""

    def __init__(self, inner: PitVault) -> None:
        self.recorder = inner.recorder
        self.watchdog = inner.watchdog
        self._inner = inner
        self._leaked = False

    def asof(self, name: str, t: datetime, **kwargs: Any) -> Any:
        if name == BARS_DATASET and not self._leaked:
            self._leaked = True
            # The leak: decide at t but read what is known at t + 1 day.
            self._inner.asof(name, t + timedelta(days=1))
        return self._inner.asof(name, t, **kwargs)


def test_future_read_at_decision_time_trips_watchdog(tmp_path, stub_engine) -> None:
    """ADVERSARIAL §1b-W1 regression: asof(t+1d) at decision t must trip."""
    recorder = InMemoryRecorder()
    pit_root = tmp_path / "pit"
    inner = _build_vault(pit_root, recorder)
    leaky = _LeakyStrategyVault(inner)
    with pytest.raises(LeakageError, match="leakage"):
        run_backtest_proven(
            AppConfig(),
            seed=42,
            pit_root=pit_root,
            bundle_dir=tmp_path / "proofs",
            vault=leaky,
            recorder=recorder,
        )
    # the watchdog observed the honest first-decision reads before tripping
    assert inner.watchdog.n_observed >= 1  # type: ignore[union-attr]


def test_honest_proven_run_passes_watchdog(tmp_path, stub_engine) -> None:
    """Control: the same proven run without the leak mints and verifies."""
    recorder = InMemoryRecorder()
    pit_root = tmp_path / "pit"
    vault = _build_vault(pit_root, recorder)
    bundle = run_backtest_proven(
        AppConfig(),
        seed=42,
        pit_root=pit_root,
        bundle_dir=tmp_path / "proofs",
        vault=vault,
        recorder=recorder,
    )
    assert bundle.data_manifest.n_reads == 1 + 2 * N_DAYS
    assert vault.watchdog is not None
    assert vault.watchdog.n_observed == 1 + 2 * N_DAYS  # type: ignore[union-attr]


def test_rogue_vault_reads_land_in_bundle(
    tmp_path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """ADVERSARIAL §1b-W2 regression: a second PitVault(root) opened mid-run
    (no recorder of its own) auto-attaches; its read is in the manifest."""
    recorder = InMemoryRecorder()
    pit_root = tmp_path / "pit"
    _build_vault(pit_root, recorder)

    def rogue_engine(bars: pl.DataFrame, weights: pl.DataFrame, config: Any) -> Any:
        rogue = PitVault(pit_root)  # no recorder/watchdog: the W2 attack
        rogue.asof(BARS_DATASET, ASOF_SENTINEL)  # silent read pre-fix
        return _stub_engine(bars, weights, config)

    monkeypatch.setattr("quant_fund.proof.runner._engine_fn", lambda _engine: rogue_engine)
    with caplog.at_level(logging.WARNING, logger="quant_fund.pit.vault"):
        bundle = run_backtest_proven(
            AppConfig(),
            seed=42,
            pit_root=pit_root,
            bundle_dir=tmp_path / "proofs",
            recorder=recorder,
        )
    assert bundle.data_manifest.n_reads == 1 + 2 * N_DAYS + 1  # + rogue read
    rogue_reads = [r for r in bundle.data_manifest.reads if r.asof_utc == ASOF_SENTINEL.isoformat()]
    # one sentinel probe from the runner (weights) + one rogue read (bars)
    assert {r.dataset for r in rogue_reads} == {WEIGHTS_DATASET, BARS_DATASET}
    assert any("auto-attached" in record.message for record in caplog.records)


class _ThreadedRogueVault:
    """Strategy-side wrapper (ADVERSARIAL R2 §1-W6 repro): on the FIRST bars
    decision read at decision time t — i.e. inside ``decision_window(t)`` —
    it spawns a BARE ``threading.Thread`` (no contextvars propagation) that
    reads ``asof(t + 1 day)`` through a rogue ``PitVault(root)``.

    Pre-fix this returned future rows with ``n_reads=0``/``n_observed=0`` —
    invisible to recorder, watchdog, and decision window.
    """

    def __init__(self, inner: PitVault, pit_root: Path) -> None:
        self.recorder = inner.recorder
        self.watchdog = inner.watchdog
        self._inner = inner
        self._pit_root = pit_root
        self._attacked = False
        self.thread_error: BaseException | None = None
        self.thread_rows: int | None = None

    def asof(self, name: str, t: datetime, **kwargs: Any) -> Any:
        if name == BARS_DATASET and not self._attacked:
            self._attacked = True

            def rogue_read() -> None:
                try:
                    rogue = PitVault(self._pit_root)  # no hooks, bare thread
                    self.thread_rows = rogue.asof(name, t + timedelta(days=1)).rows
                except BaseException as exc:  # capture what the worker sees
                    self.thread_error = exc

            worker = threading.Thread(target=rogue_read)
            worker.start()
            worker.join()  # still inside decision_window(t)
        return self._inner.asof(name, t, **kwargs)


def test_worker_thread_rogue_read_trips_and_records(
    tmp_path, stub_engine, caplog: pytest.LogCaptureFixture
) -> None:
    """ADVERSARIAL R2 §1-W6 regression: a rogue future read from a bare
    worker thread mid-proven-run now TRIPS (LeakageError in the reading
    thread, against the run's CURRENT decision window) and is RECORDED into
    the run's manifest (n_reads > 0) instead of escaping silently."""
    recorder = InMemoryRecorder()
    pit_root = tmp_path / "pit"
    inner = _build_vault(pit_root, recorder)
    rogue = _ThreadedRogueVault(inner, pit_root)
    with caplog.at_level(logging.WARNING, logger="quant_fund.pit.vault"):
        bundle = run_backtest_proven(
            AppConfig(),
            seed=42,
            pit_root=pit_root,
            bundle_dir=tmp_path / "proofs",
            vault=rogue,
            recorder=recorder,
        )
    # Trip: the future read raised LeakageError IN THE WORKER THREAD, checked
    # against the run's current decision window via the thread-visible
    # registry (pre-fix: no error, 365 future rows returned).
    assert isinstance(rogue.thread_error, LeakageError)
    assert rogue.thread_rows is None  # the read never returned data
    # Record: the read landed in the bundle's manifest anyway (recorded
    # before the watchdog tripped) — it can never escape the manifest.
    assert bundle.data_manifest.n_reads == 1 + 2 * N_DAYS + 1
    assert any("cross-thread" in record.message for record in caplog.records)


def test_worker_thread_read_without_window_is_recorded(
    tmp_path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """ADVERSARIAL R2 §1-W6: a bare-thread rogue read while no decision
    window is active still attaches via the registry and lands in the
    manifest (record-only path; the legacy decision_time -> asof fallback
    applies outside windows)."""
    recorder = InMemoryRecorder()
    pit_root = tmp_path / "pit"
    _build_vault(pit_root, recorder)
    observed: dict[str, Any] = {}

    def threaded_engine(bars: pl.DataFrame, weights: pl.DataFrame, config: Any) -> Any:
        def rogue_read() -> None:
            rogue = PitVault(pit_root)  # no hooks, bare thread, no window
            observed["rows"] = rogue.asof(BARS_DATASET, ASOF_SENTINEL).rows

        worker = threading.Thread(target=rogue_read)
        worker.start()
        worker.join()
        return _stub_engine(bars, weights, config)

    monkeypatch.setattr("quant_fund.proof.runner._engine_fn", lambda _engine: threaded_engine)
    with caplog.at_level(logging.WARNING, logger="quant_fund.pit.vault"):
        bundle = run_backtest_proven(
            AppConfig(),
            seed=42,
            pit_root=pit_root,
            bundle_dir=tmp_path / "proofs",
            recorder=recorder,
        )
    assert observed["rows"] > 0  # read happened (legacy fallback, no window)
    assert bundle.data_manifest.n_reads == 1 + 2 * N_DAYS + 1  # but recorded
    assert any("cross-thread" in record.message for record in caplog.records)


def test_proven_thread_honest_reads_pass(
    tmp_path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """ADVERSARIAL R2 §1-W6 blessed path: ``run_context.proven_thread``
    propagates the proven-run context into the worker — honest reads through
    it are recorded, watchdog-clean, and need no cross-thread fallback."""
    recorder = InMemoryRecorder()
    pit_root = tmp_path / "pit"
    _build_vault(pit_root, recorder)
    observed: dict[str, Any] = {}

    def honest_engine(bars: pl.DataFrame, weights: pl.DataFrame, config: Any) -> Any:
        def honest_read() -> None:
            observed["context_is_proven"] = run_context.context_is_proven()
            observed["recorder_is_run"] = run_context.active_recorder() is recorder
            observed["rows"] = PitVault(pit_root).asof(BARS_DATASET, ASOF_SENTINEL).rows

        worker = run_context.proven_thread(honest_read)
        worker.start()
        worker.join()
        return _stub_engine(bars, weights, config)

    monkeypatch.setattr("quant_fund.proof.runner._engine_fn", lambda _engine: honest_engine)
    with caplog.at_level(logging.WARNING, logger="quant_fund.pit.vault"):
        bundle = run_backtest_proven(
            AppConfig(),
            seed=42,
            pit_root=pit_root,
            bundle_dir=tmp_path / "proofs",
            recorder=recorder,
        )
    assert observed["context_is_proven"] is True  # contextvars propagated
    assert observed["recorder_is_run"] is True
    assert observed["rows"] > 0
    assert bundle.data_manifest.n_reads == 1 + 2 * N_DAYS + 1
    assert not any("cross-thread" in record.message for record in caplog.records)


def test_proven_thread_inside_decision_window_passes(
    tmp_path, stub_engine, caplog: pytest.LogCaptureFixture
) -> None:
    """proven_thread propagates the active decision window too: an honest
    asof(t) read from a worker inside ``decision_window(t)`` passes the
    watchdog (checked against t, which the worker inherits)."""
    recorder = InMemoryRecorder()
    pit_root = tmp_path / "pit"
    inner = _build_vault(pit_root, recorder)
    observed: dict[str, Any] = {}

    class WindowedThreadVault:
        def __init__(self, wrapped: PitVault) -> None:
            self.recorder = wrapped.recorder
            self.watchdog = wrapped.watchdog
            self._wrapped = wrapped
            self._spawned = False

        def asof(self, name: str, t: datetime, **kwargs: Any) -> Any:
            if name == BARS_DATASET and not self._spawned:
                self._spawned = True

                def honest_read() -> None:
                    observed["decision_time"] = run_context.current_decision_time()
                    observed["rows"] = PitVault(pit_root).asof(name, t).rows

                worker = run_context.proven_thread(honest_read)
                worker.start()
                worker.join()
            return self._wrapped.asof(name, t, **kwargs)

    with caplog.at_level(logging.WARNING, logger="quant_fund.pit.vault"):
        bundle = run_backtest_proven(
            AppConfig(),
            seed=42,
            pit_root=pit_root,
            bundle_dir=tmp_path / "proofs",
            vault=WindowedThreadVault(inner),
            recorder=recorder,
        )
    first_decision = min(
        datetime.fromisoformat(r.asof_utc)
        for r in bundle.data_manifest.reads
        if r.dataset == BARS_DATASET and r.asof_utc != ASOF_SENTINEL.isoformat()
    )
    assert observed["decision_time"] == first_decision  # window propagated
    assert observed["rows"] > 0  # honest read passed the watchdog
    assert bundle.data_manifest.n_reads == 1 + 2 * N_DAYS + 1
    assert not any("cross-thread" in record.message for record in caplog.records)


def test_sequential_proven_runs_no_interference(tmp_path, stub_engine) -> None:
    """The thread-visible registry is scoped to each proven run: after a run
    exits, stray reads attach to nothing, and a second proven run records
    exactly its own reads."""
    pit_root = tmp_path / "pit"
    recorder1 = InMemoryRecorder()
    _build_vault(pit_root, recorder1)
    bundle1 = run_backtest_proven(
        AppConfig(),
        seed=42,
        pit_root=pit_root,
        bundle_dir=tmp_path / "proofs1",
        recorder=recorder1,
    )
    n1 = bundle1.data_manifest.n_reads
    assert n1 == 1 + 2 * N_DAYS

    # Between runs: no active run on any thread — a stray rogue read must
    # attach to NOTHING (registry restored on exit).
    assert run_context.active_recorder() is None
    assert run_context.current_decision_time() is None
    PitVault(pit_root).asof(BARS_DATASET, ASOF_SENTINEL)
    assert len(recorder1.reads) == n1  # run 1's recorder untouched

    recorder2 = InMemoryRecorder()
    bundle2 = run_backtest_proven(
        AppConfig(),
        seed=43,
        pit_root=pit_root,
        bundle_dir=tmp_path / "proofs2",
        recorder=recorder2,
    )
    assert bundle2.data_manifest.n_reads == n1  # exactly its own reads
    assert run_context.active_recorder() is None
    assert run_context.current_decision_time() is None


def test_signed_bundle_replay_verifies(
    tmp_path, stub_engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADVERSARIAL §2-R regression: pristine signed bundle passes --replay."""
    from quant_fund.proof.verify import verify_bundle

    monkeypatch.setenv("PROOFCORE_SIGNING_KEY", "hardening-ci-key")
    recorder = InMemoryRecorder()
    pit_root = tmp_path / "pit"
    _build_vault(pit_root, recorder)
    bundle_dir = tmp_path / "proofs"
    bundle = run_backtest_proven(
        AppConfig(),
        seed=42,
        pit_root=pit_root,
        bundle_dir=bundle_dir,
        signer=HmacSha256Signer.from_env(),
        recorder=recorder,
    )
    bundle_path = bundle_dir / "bundles" / f"{bundle.bundle_id}.json"
    result = verify_bundle(bundle_path, bundle_dir=bundle_dir, replay=True, pit_root=pit_root)
    assert result.ok, result.reasons


def test_signed_bundle_tamper_still_fails(
    tmp_path, stub_engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADVERSARIAL §2-E regression guard: flipping seed breaks self-hash + HMAC."""
    import json

    from quant_fund.proof.verify import verify_bundle

    monkeypatch.setenv("PROOFCORE_SIGNING_KEY", "hardening-ci-key")
    recorder = InMemoryRecorder()
    pit_root = tmp_path / "pit"
    _build_vault(pit_root, recorder)
    bundle_dir = tmp_path / "proofs"
    bundle = run_backtest_proven(
        AppConfig(),
        seed=42,
        pit_root=pit_root,
        bundle_dir=bundle_dir,
        signer=HmacSha256Signer.from_env(),
        recorder=recorder,
    )
    bundle_path = bundle_dir / "bundles" / f"{bundle.bundle_id}.json"
    payload = json.loads(bundle_path.read_bytes())
    payload["seed"] = 777
    bundle_path.write_bytes(json.dumps(payload).encode())
    result = verify_bundle(bundle_path, bundle_dir=bundle_dir)
    assert not result.ok
    assert "bundle_id:self_hash_mismatch" in result.reasons
    assert "signature:hmac_mismatch" in result.reasons
