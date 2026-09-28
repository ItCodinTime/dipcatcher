"""Runtime leakage watchdog — monkeypatch-free interposition (DESIGN.md §6.3).

``LeakageWatchdog`` is a constructor-injected collaborator of
``PitVault(watchdog=...)``: on every ``asof()`` read the vault reports the
``DataAccessRecord`` plus the caller's decision time, and the watchdog asserts
``max known_at of returned rows <= decision_time``. Because interposition is
via the constructor, it cannot be bypassed by import-order tricks; code that
never touches the vault is caught by LH009 + the W5 grep gate instead.

Channel contract: the vault conveys the returned frame's maximum ``known_at``
via ``DataAccessRecord.params["max_known_at"]`` (ISO-8601; params are
str-coerced at the recorder boundary per contracts). In ``strict`` mode a read
without that watermark fails closed — an unverifiable read is treated as a
potential leak.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from quant_fund.proofcore.contracts import DataAccessRecord
from quant_fund.proofcore.contracts import LeakageError as ContractsLeakageError
from quant_fund.schemas.errors import LeakageError as SchemasLeakageError

MAX_KNOWN_AT_PARAM = "max_known_at"


class LeakageError(ContractsLeakageError, SchemasLeakageError):
    """Watchdog trip or error-severity scan finding (package-boundary adapter).

    Subclasses both ``contracts.LeakageError`` (PROOFCORE catch root) and
    ``quant_fund.schemas.errors.LeakageError`` so existing
    ``except PointInTimeError``/``except LeakageError`` clauses keep working
    (DESIGN.md §8.3).
    """


class WatchdogProtocol(Protocol):
    """Constructor-injected observer contract consumed by ``PitVault``."""

    def observe(self, read: DataAccessRecord, decision_time: datetime) -> None: ...


def _require_aware(dt: datetime, *, what: str) -> datetime:
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise LeakageError(f"{what} must be tz-aware; fail-closed on naive timestamps")
    return dt


class LeakageWatchdog:
    """Asserts every observed vault read is PIT-clean; tracks a watermark.

    Trip = :class:`LeakageError` carrying dataset, asof, and the offending
    max ``known_at``. Also maintains a per-run ``max_known_at_consumed``
    watermark exported into the proof bundle's data manifest (defense in
    depth on top of the vault filter).
    """

    def __init__(self, *, strict: bool = True) -> None:
        self._strict = strict
        self._watermark: datetime | None = None
        self._n_observed: int = 0

    @property
    def strict(self) -> bool:
        return self._strict

    @property
    def n_observed(self) -> int:
        return self._n_observed

    def observe(self, read: DataAccessRecord, decision_time: datetime) -> None:
        decision_time = _require_aware(decision_time, what="decision_time")
        raw = read.params.get(MAX_KNOWN_AT_PARAM)
        if raw is None:
            if self._strict:
                raise LeakageError(
                    f"unverifiable read of dataset {read.dataset!r} at "
                    f"{read.asof_utc}: DataAccessRecord lacks "
                    f"params[{MAX_KNOWN_AT_PARAM!r}]; fail-closed"
                )
            return
        try:
            max_known_at = datetime.fromisoformat(raw)
        except (ValueError, TypeError) as exc:
            raise LeakageError(
                f"malformed {MAX_KNOWN_AT_PARAM} watermark {raw!r} on dataset "
                f"{read.dataset!r}; fail-closed"
            ) from exc
        _require_aware(max_known_at, what=MAX_KNOWN_AT_PARAM)
        self._n_observed += 1
        if max_known_at > decision_time:
            raise LeakageError(
                f"leakage: dataset {read.dataset!r} read at {read.asof_utc} "
                f"returned rows with known_at up to {max_known_at.isoformat()} "
                f"> decision_time {decision_time.isoformat()}"
            )
        if self._watermark is None or max_known_at > self._watermark:
            self._watermark = max_known_at

    def watermark(self) -> datetime | None:
        """Max known_at consumed so far this run (None before any read)."""
        return self._watermark
