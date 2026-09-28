"""Runtime IO guard — interposition on raw reads inside proven windows
(WAVE2.md §6).

Wave 1's watchdog covers reads that go THROUGH ``PitVault.asof``; it cannot
see strategy code that bypasses the vault entirely with a raw
``pl.read_parquet`` / ``pd.read_parquet`` / ``open()`` — the "rogue second
vault" class. ``install_io_guard`` closes that class at the call-site level
by monkey-interposing those three entry points for the duration of a proven
run.

Activation rule (binding): the guard intercepts ONLY while a proven decision
window is active, i.e. ``proofcore.run_context.current_decision_time() is not
None``. That accessor falls back to the R2 thread-visible registry, so reads
from bare ``threading.Thread`` workers (which carry no contextvars context)
are covered too. Outside a window — including everywhere outside proven runs —
the wrappers are a single ``is None`` check away from the original function:
zero interception, no behavior change.

Modes:

- ``"enforce"`` (proven-runner default): a raw read of a non-allowlisted
  path inside an active window raises :class:`LeakageError`. A vault-mapped
  path is never read raw — it is routed through the mapped vault's
  ``asof(name, decision_time)`` so the read lands in the recorder/watchdog.
- ``"redirect"``: vault-mapped paths route through ``vault.asof``; unknown
  paths warn (``IOGuardWarning``), are recorded, and proceed.
- ``"audit"``: record everything, raise nothing, redirect nothing.

The guard is re-entrant (nested ``install_io_guard`` yields the already
installed guard; interposition is removed only when the outermost exit runs)
and thread-safe (module lock around install/teardown; per-guard lock around
the record log). Interposition is always fully removed on exit, including on
exceptions.

Layering (DESIGN.md §1.3): imports ``leakage.watchdog`` (same package) and
``proofcore.run_context`` (layer 1) only; polars/pandas are imported lazily
at install time so importing this module stays cheap and side-effect free.
"""

from __future__ import annotations

import builtins
import functools
import io
import threading
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from quant_fund.leakage.watchdog import LeakageError
from quant_fund.proofcore import run_context

__all__ = [
    "GUARD_MODES",
    "GuardRecord",
    "IOGuard",
    "IOGuardWarning",
    "install_io_guard",
]

GuardMode = Literal["redirect", "enforce", "audit"]
GUARD_MODES: tuple[str, ...] = ("redirect", "enforce", "audit")

# Interposed function labels (for GuardRecord.function).
_PL_READ_PARQUET = "polars.read_parquet"
_PD_READ_PARQUET = "pandas.read_parquet"
_OPEN = "open"


class IOGuardWarning(UserWarning):
    """Redirect-mode warning: a raw read of an unknown path proceeded."""


@dataclass(frozen=True)
class GuardRecord:
    """One intercepted (or would-be-intercepted) read inside a window."""

    function: str  # e.g. "polars.read_parquet" | "pandas.read_parquet" | "open"
    path: str | None  # raw path argument as a string; None for path-less sources
    mode: str  # guard mode in effect
    action: str  # "redirected" | "warned" | "blocked" | "audited" | "allowlisted"
    decision_time: datetime | None
    thread_id: int


def _resolve(path: Path) -> Path:
    """Best-effort absolute, symlink-resolved path (never raises)."""
    try:
        return path.expanduser().resolve()
    except OSError:
        return path.expanduser().absolute()


def _extract_path(source: Any) -> Path | None:
    """Pull a filesystem path out of a read call's first argument.

    ``str``/``os.PathLike`` sources yield a path; file-like objects, lists of
    sources, and anything else yield None (treated as unknown — fail-closed
    in enforce mode).
    """
    if isinstance(source, (str, Path)):
        return Path(source)
    return None


def _is_read_mode(mode: str) -> bool:
    """True when an ``open()`` mode reads without writing/appending."""
    return not any(flag in mode for flag in ("w", "a", "x", "+"))


class IOGuard:
    """Configuration + record log for one installed IO guard.

    Constructed by :func:`install_io_guard`; not meant to be instantiated
    directly by caller code (but harmless — interception happens only via the
    module-level installed guard while a window is active).
    """

    def __init__(
        self,
        *,
        mode: GuardMode,
        vault_map: Mapping[str | Path, Any] | None = None,
        allowlist: tuple[str | Path, ...] = (),
    ) -> None:
        if mode not in GUARD_MODES:
            raise ValueError(f"unknown IO guard mode {mode!r}; expected one of {GUARD_MODES}")
        self._mode: GuardMode = mode
        # Resolved path-prefix -> vault (longest prefix wins at lookup time).
        self._vault_roots: tuple[tuple[Path, Any], ...] = tuple(
            sorted(
                ((_resolve(Path(prefix)), vault) for prefix, vault in (vault_map or {}).items()),
                key=lambda item: len(item[0].parts),
                reverse=True,
            )
        )
        self._allowlist: tuple[Path, ...] = tuple(
            _resolve(Path(entry)) for entry in allowlist
        )
        self._records: list[GuardRecord] = []
        self._lock = threading.Lock()

    @property
    def mode(self) -> GuardMode:
        return self._mode

    @property
    def records(self) -> tuple[GuardRecord, ...]:
        """Thread-safe snapshot of everything the guard has seen/done."""
        with self._lock:
            return tuple(self._records)

    # -- decision helpers -----------------------------------------------------

    def _record(self, record: GuardRecord) -> None:
        with self._lock:
            self._records.append(record)

    def _vault_for(self, path: Path) -> tuple[Any, str] | None:
        """The mapped vault + dataset name for ``path``, or None.

        The dataset name is the path relative to the registered vault root,
        posix-style, without the file suffix (e.g. ``<root>/main/part.parquet``
        under root ``<root>`` maps to dataset ``"main/part"``).
        """
        resolved = _resolve(path)
        for root, vault in self._vault_roots:
            if resolved == root or resolved.is_relative_to(root):
                rel = resolved.relative_to(root)
                name = rel.with_suffix("").as_posix()
                return vault, name
        return None

    def _is_allowlisted(self, path: Path) -> bool:
        resolved = _resolve(path)
        for entry in self._allowlist:
            if resolved == entry or resolved.is_relative_to(entry):
                return True
        return False

    # -- interception entry points (called by the wrappers) -------------------

    def handle_read(
        self,
        function: str,
        path: Path | None,
        proceed: Callable[[], Any],
    ) -> Any:
        """Apply the mode's policy to one raw read inside an active window.

        ``proceed`` is a thunk performing the original call. Returns the
        (possibly redirected) read result; raises ``LeakageError`` in enforce
        mode for non-allowlisted, non-vault-mapped reads.
        """
        decision_time = run_context.current_decision_time()
        thread_id = threading.get_ident()
        path_str = None if path is None else str(path)

        def log(action: str) -> None:
            self._record(
                GuardRecord(
                    function=function,
                    path=path_str,
                    mode=self._mode,
                    action=action,
                    decision_time=decision_time,
                    thread_id=thread_id,
                )
            )

        if path is not None and self._is_allowlisted(path):
            log("allowlisted")
            return proceed()

        mapped = self._vault_for(path) if path is not None else None
        if self._mode == "audit":
            log("audited")
            return proceed()
        if mapped is not None:
            vault, name = mapped
            log("redirected")
            return vault.asof(name, decision_time)
        if self._mode == "enforce":
            log("blocked")
            raise LeakageError(
                f"io_guard: raw {function} read of {path_str!r} inside a proven "
                f"decision window (decision_time={decision_time.isoformat() if decision_time else None}) "
                "is not allowlisted and not vault-mapped — reads inside proven "
                "windows must go through PitVault.asof (mode='enforce')"
            )
        # redirect mode: unknown path -> warn + record + proceed.
        log("warned")
        warnings.warn(
            f"io_guard: raw {function} read of {path_str!r} inside a proven "
            "decision window is not vault-mapped; proceeding (mode='redirect')",
            IOGuardWarning,
            stacklevel=3,
        )
        return proceed()


# ---------------------------------------------------------------------------
# Module-level interposition machinery (re-entrant, thread-safe).
# ---------------------------------------------------------------------------

_STATE_LOCK = threading.RLock()
_ACTIVE_GUARD: IOGuard | None = None
_NESTING = 0
_ORIGINALS: dict[str, Any] = {}


def _intercepting_guard() -> IOGuard | None:
    """The guard to apply to THIS call, or None for zero interception.

    Active iff a guard is installed AND a proven decision window is active.
    ``current_decision_time()`` falls back to the R2 thread-visible registry,
    so bare worker threads are covered (ADVERSARIAL R2 §1-W6).
    """
    guard = _ACTIVE_GUARD
    if guard is None:
        return None
    if run_context.current_decision_time() is None:
        return None
    return guard


def _wrap_reader(function: str, original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapper(source: Any, *args: Any, **kwargs: Any) -> Any:
        guard = _intercepting_guard()
        if guard is None:
            return original(source, *args, **kwargs)
        return guard.handle_read(
            function, _extract_path(source), lambda: original(source, *args, **kwargs)
        )

    return wrapper


def _wrap_open(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapper(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        guard = _intercepting_guard()
        # Only read-mode opens are interposed; writes/appends pass through.
        if guard is None or not isinstance(mode, str) or not _is_read_mode(mode):
            return original(file, mode, *args, **kwargs)
        return guard.handle_read(
            _OPEN, _extract_path(file), lambda: original(file, mode, *args, **kwargs)
        )

    return wrapper


def _install_hooks() -> None:
    """Monkey-interpose the three read entry points (under _STATE_LOCK)."""
    import pandas as pd
    import polars as pl

    _ORIGINALS[_PL_READ_PARQUET] = pl.read_parquet
    _ORIGINALS[_PD_READ_PARQUET] = pd.read_parquet
    _ORIGINALS["builtins.open"] = builtins.open
    _ORIGINALS["io.open"] = io.open
    pl.read_parquet = _wrap_reader(_PL_READ_PARQUET, _ORIGINALS[_PL_READ_PARQUET])  # type: ignore[assignment]
    pd.read_parquet = _wrap_reader(_PD_READ_PARQUET, _ORIGINALS[_PD_READ_PARQUET])  # type: ignore[assignment]
    open_wrapper = _wrap_open(_ORIGINALS["builtins.open"])
    builtins.open = open_wrapper  # type: ignore[assignment]
    io.open = open_wrapper  # type: ignore[assignment]


def _remove_hooks() -> None:
    """Restore every interposed function to its original (under _STATE_LOCK)."""
    import pandas as pd
    import polars as pl

    pl.read_parquet = _ORIGINALS.pop(_PL_READ_PARQUET)  # type: ignore[assignment]
    pd.read_parquet = _ORIGINALS.pop(_PD_READ_PARQUET)  # type: ignore[assignment]
    builtins.open = _ORIGINALS.pop("builtins.open")  # type: ignore[assignment]
    io.open = _ORIGINALS.pop("io.open")  # type: ignore[assignment]


@contextmanager
def install_io_guard(
    mode: GuardMode,
    *,
    vault_map: Mapping[str | Path, Any] | None = None,
    allowlist: tuple[str | Path, ...] = (),
) -> Iterator[IOGuard]:
    """Install the IO guard for the duration of the context.

    Re-entrant: a nested ``install_io_guard`` yields the ALREADY INSTALLED
    guard (first install wins; the inner config is ignored) and interposition
    is removed only when the outermost context exits — even on exception.
    While installed, interception happens ONLY inside an active proven
    decision window (``run_context.current_decision_time() is not None``);
    outside any window the wrapped functions behave exactly as the originals.
    """
    global _ACTIVE_GUARD, _NESTING
    with _STATE_LOCK:
        if _ACTIVE_GUARD is not None:
            _NESTING += 1
            guard = _ACTIVE_GUARD
        else:
            guard = IOGuard(mode=mode, vault_map=vault_map, allowlist=allowlist)
            _install_hooks()
            _ACTIVE_GUARD = guard
            _NESTING = 1
    try:
        yield guard
    finally:
        with _STATE_LOCK:
            _NESTING -= 1
            if _NESTING == 0:
                try:
                    _remove_hooks()
                finally:
                    _ACTIVE_GUARD = None
