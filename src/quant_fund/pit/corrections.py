"""Bitemporal restatement semantics (DESIGN.md §4.2).

Corrections/restatements are **appends**, never edits: a restated record is
written with its corrected payload, the same ``event_time``, and a **new**
``known_at`` = the time the correction became known. Old versions are retained
forever. ``asof(t)`` returns, per ``(security_id, event_time)`` key, the
version with the greatest ``known_at <= t`` (``LATEST_KNOWN``) or the earliest
(``STRICT_FIRST`` — used by seeded-leak tests to prove restatements are inert
before publication).
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import cast

import polars as pl

from quant_fund.proofcore.contracts import VaultError

EVENT_TIME_COL = "event_time"
KNOWN_AT_COL = "known_at"
SECURITY_ID_COL = "security_id"


class RestatementPolicy(enum.Enum):
    """Version-selection policy for asof() reads (DESIGN.md §4.2)."""

    LATEST_KNOWN = "latest_known"
    STRICT_FIRST = "strict_first"


def key_columns(*, security_level: bool) -> list[str]:
    """Bitemporal key: (security_id, event_time) for security-level datasets."""
    if security_level:
        return [SECURITY_ID_COL, EVENT_TIME_COL]
    return [EVENT_TIME_COL]


def require_pit_frame(frame: pl.DataFrame, *, security_level: bool) -> None:
    """``require_pit_columns``-style append validation (DESIGN.md §13 phase 1).

    Fail-closed: missing/naive/null PIT columns are contract violations.
    """
    missing = [c for c in (EVENT_TIME_COL, KNOWN_AT_COL) if c not in frame.columns]
    if missing:
        raise VaultError(f"append missing PIT columns: {missing}")
    if security_level and SECURITY_ID_COL not in frame.columns:
        raise VaultError("security-level dataset append missing 'security_id'")
    for col in (EVENT_TIME_COL, KNOWN_AT_COL):
        dtype = frame.schema[col]
        if not isinstance(dtype, pl.Datetime):
            raise VaultError(f"column {col!r} must be Datetime, got {dtype}")
        if dtype.time_zone is None:
            raise VaultError(f"column {col!r} must be timezone-aware (UTC)")
    if frame.height == 0:
        raise VaultError("refusing empty append — a vault part must carry rows")
    if frame[EVENT_TIME_COL].null_count() or frame[KNOWN_AT_COL].null_count():
        raise VaultError("null event_time/known_at is unobservable — fail closed")
    if security_level and frame[SECURITY_ID_COL].null_count():
        raise VaultError("null security_id in security-level append — fail closed")
    # Two rows carrying the same (key, known_at) are ambiguous versions of one
    # publication instant; selection resolves them by arrival order, silently
    # shadowing the loser. Fail closed instead of picking a winner.
    dup_cols = [*key_columns(security_level=security_level), KNOWN_AT_COL]
    if frame.select(dup_cols).is_duplicated().any():
        raise VaultError(
            "duplicate (key, known_at) rows in one append — ambiguous version, fail closed"
        )


def normalize_pit_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Cast PIT columns to the vault storage dtype Datetime(us, UTC) (§4.1)."""
    return frame.with_columns(
        pl.col(EVENT_TIME_COL).dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC")),
        pl.col(KNOWN_AT_COL).dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC")),
    )


def prepare_correction(corrected: pl.DataFrame, *, known_at: datetime) -> pl.DataFrame:
    """Stamp a restatement with its explicit ``known_at`` publication time.

    Any ``known_at`` column already present is overridden: the caller's
    ``known_at`` is the moment the correction became known (§4.2).
    """
    if known_at.tzinfo is None or known_at.tzinfo.utcoffset(known_at) is None:
        raise VaultError("restate known_at must be timezone-aware (UTC)")
    return corrected.with_columns(pl.lit(known_at).alias(KNOWN_AT_COL))


def select_asof(
    frame: pl.LazyFrame,
    t: datetime,
    *,
    key_cols: list[str],
    policy: RestatementPolicy,
    columns: list[str] | None = None,
) -> pl.DataFrame:
    """The vault filter: latest/earliest known version per key with known_at <= t.

    The ``known_at <= t`` filter is applied BEFORE any caller code sees the
    frame; per-key selection happens inside the scan with projection pushdown
    (DESIGN.md §4.6). Output is sorted by key columns for byte-stable content
    hashes.
    """
    if t.tzinfo is None or t.tzinfo.utcoffset(t) is None:
        raise VaultError("asof timestamp must be timezone-aware (UTC)")
    needed = list(key_cols) + [KNOWN_AT_COL]
    if columns is not None:
        needed = list(dict.fromkeys(needed + list(columns)))
        scan = frame.select(needed)
    else:
        scan = frame
    available = scan.filter(pl.col(KNOWN_AT_COL) <= pl.lit(t)).sort(KNOWN_AT_COL)
    if policy is RestatementPolicy.LATEST_KNOWN:
        sliced = available.group_by(key_cols, maintain_order=True).last()
    elif policy is RestatementPolicy.STRICT_FIRST:
        sliced = available.group_by(key_cols, maintain_order=True).first()
    else:  # pragma: no cover - enum is closed
        raise VaultError(f"unknown restatement policy: {policy!r}")
    try:
        return sliced.sort(list(key_cols) + [KNOWN_AT_COL]).collect()
    except pl.exceptions.ColumnNotFoundError as exc:
        raise VaultError(f"asof projection references unknown column: {exc}") from exc


def frame_span(frame: pl.DataFrame) -> tuple[str, str, str, str]:
    """(min_known_at, max_known_at, min_event_time, max_event_time) as ISO strings."""
    return (
        cast(datetime, frame[KNOWN_AT_COL].min()).isoformat(),
        cast(datetime, frame[KNOWN_AT_COL].max()).isoformat(),
        cast(datetime, frame[EVENT_TIME_COL].min()).isoformat(),
        cast(datetime, frame[EVENT_TIME_COL].max()).isoformat(),
    )
