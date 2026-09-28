"""Active proven-run provenance context (contextvars; ADVERSARIAL W1/W2 fix).

While ``run_backtest_proven`` executes, the run's recorder, watchdog, and
current decision time live in context variables so that:

- ANY ``PitVault`` touched by strategy code — including a rogue second
  ``PitVault(root)`` constructed mid-run outside the runner (ADVERSARIAL
  §1b-W2) — auto-attaches to the active recorder/watchdog at read time, so
  its reads land in the proof bundle's data manifest and under the watchdog
  instead of bypassing provenance silently;
- the vault's watchdog observation carries the runner's DECISION time
  (ADVERSARIAL §1b-W1), not the read's ``asof`` argument, so a strategy
  reading ``asof(t + 1d)`` while deciding at ``t`` trips the watchdog.

Threading (ADVERSARIAL R2 §1-W6 fix): contextvars do NOT propagate into
``threading.Thread`` workers, so a vault read from a worker thread used to
escape both the recorder and the watchdog. Two defenses:

- ``proven_thread`` / ``PropagatingThread`` run the target inside
  ``contextvars.copy_context()``, so strategy code has a blessed way to
  thread under a proven run with full context propagation;
- a module-level, lock-guarded REGISTRY mirrors the active run's
  recorder/watchdog and current decision window. The accessors below fall
  back to it when the calling thread carries no proven-run context, so a
  vault read from ANY thread during a proven run still attaches to the run's
  recorder/watchdog (with a cross-thread warning from the vault) and is
  checked against the run's CURRENT decision window — a future read raises
  ``LeakageError`` in the reading thread, and a read outside any decision
  window fails closed with ``VaultError`` there, instead of escaping
  silently.

Layering (DESIGN.md §1.3, layer 1): stdlib + contracts only; pit, proof, and
leakage may all import this module.
"""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Any

from quant_fund.proofcore.contracts import ProofError

__all__ = [
    "PropagatingThread",
    "active_recorder",
    "active_watchdog",
    "context_is_proven",
    "current_decision_time",
    "decision_window",
    "proven_run",
    "proven_thread",
]

_active_recorder: ContextVar[Any | None] = ContextVar("proofcore_active_recorder", default=None)
_active_watchdog: ContextVar[Any | None] = ContextVar("proofcore_active_watchdog", default=None)
_active_decision_time: ContextVar[datetime | None] = ContextVar(
    "proofcore_active_decision_time", default=None
)


class _RunRegistry:
    """Thread-visible mirror of the active proven run (ADVERSARIAL R2 §1-W6).

    Written only by ``proven_run``/``decision_window`` (on the thread that
    drives the run) and read by vault reads on ANY thread whose contextvars
    carry no proven-run context. Guarded by a lock; proven runs are driven
    sequentially, so save/restore on entry/exit suffices for nesting.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._recorder: Any | None = None
        self._watchdog: Any | None = None
        self._decision_time: datetime | None = None

    def snapshot(self) -> tuple[Any | None, Any | None, datetime | None]:
        with self._lock:
            return self._recorder, self._watchdog, self._decision_time

    def set_run(self, recorder: Any | None, watchdog: Any | None) -> tuple[Any, Any, Any]:
        with self._lock:
            previous = (self._recorder, self._watchdog, self._decision_time)
            self._recorder = recorder
            self._watchdog = watchdog
            self._decision_time = None
            return previous

    def restore_run(self, previous: tuple[Any, Any, Any]) -> None:
        with self._lock:
            self._recorder, self._watchdog, self._decision_time = previous

    def set_decision_time(self, decision_time: datetime | None) -> datetime | None:
        with self._lock:
            previous = self._decision_time
            self._decision_time = decision_time
            return previous

    def restore_decision_time(self, previous: datetime | None) -> None:
        with self._lock:
            self._decision_time = previous


_REGISTRY = _RunRegistry()


def active_recorder() -> Any | None:
    """The recorder of the in-flight proven run, if any.

    Falls back to the thread-visible registry when this thread's context
    carries no proven run (worker threads; ADVERSARIAL R2 §1-W6).
    """
    recorder = _active_recorder.get()
    if recorder is None:
        recorder, _, _ = _REGISTRY.snapshot()
    return recorder


def active_watchdog() -> Any | None:
    """The watchdog of the in-flight proven run, if any (registry fallback)."""
    watchdog = _active_watchdog.get()
    if watchdog is None:
        _, watchdog, _ = _REGISTRY.snapshot()
    return watchdog


def current_decision_time() -> datetime | None:
    """The decision time the proven run is currently deciding at, if set.

    Falls back to the registry's current decision window for threads that
    carry no proven-run context of their own.
    """
    decision_time = _active_decision_time.get()
    if decision_time is None:
        _, _, decision_time = _REGISTRY.snapshot()
    return decision_time


def context_is_proven() -> bool:
    """True when THIS thread's own context carries an active proven run.

    False for worker threads that only see the run through the registry
    fallback — the vault uses this to word its cross-thread attach warning.
    """
    return _active_recorder.get() is not None


@contextmanager
def proven_run(recorder: Any, watchdog: Any | None = None) -> Iterator[None]:
    """Mark the current context as an active proven run.

    The active hooks take precedence over private vault hooks. A decision
    window is mandatory for each read; this context does not implement the
    still-unavailable proven backtest orchestrator. The run is also mirrored
    into the thread-visible registry so worker threads without this context
    still attach (ADVERSARIAL R2 §1-W6).
    """
    if recorder is None or watchdog is None:
        raise ProofError("proven context requires both recorder and watchdog")
    token_clock = _active_decision_time.set(None)
    token_recorder = _active_recorder.set(recorder)
    token_watchdog = _active_watchdog.set(watchdog)
    previous = _REGISTRY.set_run(recorder, watchdog)
    try:
        yield
    finally:
        _REGISTRY.restore_run(previous)
        _active_watchdog.reset(token_watchdog)
        _active_recorder.reset(token_recorder)
        _active_decision_time.reset(token_clock)


@contextmanager
def decision_window(decision_time: datetime) -> Iterator[None]:
    """Mark the current context as deciding at ``decision_time``.

    Vault reads inside the window are checked by the watchdog against this
    decision time, not against the read's ``asof`` watermark. The window is
    also mirrored into the thread-visible registry so reads from threads
    without this context are checked against the run's CURRENT decision
    window (ADVERSARIAL R2 §1-W6).
    """
    if decision_time.tzinfo is None or decision_time.utcoffset() is None:
        raise ProofError("decision time must be timezone-aware")
    token = _active_decision_time.set(decision_time)
    previous = _REGISTRY.set_decision_time(decision_time)
    try:
        yield
    finally:
        _REGISTRY.restore_decision_time(previous)
        _active_decision_time.reset(token)


class PropagatingThread(threading.Thread):
    """``threading.Thread`` that runs its target inside a copy of the
    CREATING thread's contextvars context (ADVERSARIAL R2 §1-W6).

    Inside a proven run this propagates the recorder/watchdog/decision-window
    context into the worker, so its vault reads are recorded and
    watchdog-checked exactly as if made on the driving thread.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Snapshot the creating thread's contextvars context, then wrap the
        # target so the worker runs it inside that snapshot. Wrapping keeps
        # threading.Thread.run responsible for invocation and cleanup —
        # no access to its private _target/_args/_kwargs fields.
        self._proven_context = contextvars.copy_context()
        positional_target = len(args) > 1
        target: Callable[..., Any] | None = (
            kwargs.get("target") if not positional_target else args[1]
        )
        if target is not None:
            ctx = self._proven_context

            def wrapped(*a: Any, **k: Any) -> Any:
                return ctx.run(target, *a, **k)

            if positional_target:
                args = (args[0], wrapped, *args[2:])
            else:
                kwargs["target"] = wrapped
        super().__init__(*args, **kwargs)


def proven_thread(target: Callable[..., Any], *args: Any, **kwargs: Any) -> PropagatingThread:
    """Return an UNSTARTED :class:`PropagatingThread` for ``target``.

    The blessed way to run strategy work in a worker thread under a proven
    run: ``proven_thread(fn, *args, **kwargs).start()`` propagates the active
    proven-run context into the worker, unlike a bare ``threading.Thread``
    (whose reads would attach cross-thread via the registry fallback instead,
    with a warning).
    """
    return PropagatingThread(target=target, args=args, kwargs=kwargs)
