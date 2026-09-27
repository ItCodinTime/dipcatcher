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

Layering (DESIGN.md §1.3, layer 1): stdlib + contracts only; pit, proof, and
leakage may all import this module.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Any

_active_recorder: ContextVar[Any | None] = ContextVar("proofcore_active_recorder", default=None)
_active_watchdog: ContextVar[Any | None] = ContextVar("proofcore_active_watchdog", default=None)
_active_decision_time: ContextVar[datetime | None] = ContextVar(
    "proofcore_active_decision_time", default=None
)


def active_recorder() -> Any | None:
    """The recorder of the in-flight proven run, if any."""
    return _active_recorder.get()


def active_watchdog() -> Any | None:
    """The watchdog of the in-flight proven run, if any."""
    return _active_watchdog.get()


def current_decision_time() -> datetime | None:
    """The decision time the proven run is currently deciding at, if set."""
    return _active_decision_time.get()


@contextmanager
def proven_run(recorder: Any, watchdog: Any | None = None) -> Iterator[None]:
    """Mark the current context as an active proven run.

    Vaults that read while this context is active and carry no recorder of
    their own auto-attach to ``recorder``/``watchdog`` (with a warning) so
    their reads are recorded.
    """
    token_recorder = _active_recorder.set(recorder)
    token_watchdog = _active_watchdog.set(watchdog)
    try:
        yield
    finally:
        _active_watchdog.reset(token_watchdog)
        _active_recorder.reset(token_recorder)


@contextmanager
def decision_window(decision_time: datetime) -> Iterator[None]:
    """Mark the current context as deciding at ``decision_time``.

    Vault reads inside the window are checked by the watchdog against this
    decision time, not against the read's ``asof`` watermark.
    """
    token = _active_decision_time.set(decision_time)
    try:
        yield
    finally:
        _active_decision_time.reset(token)
