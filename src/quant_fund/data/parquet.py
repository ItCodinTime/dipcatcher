"""Role-aware Parquet reads at the repository data boundary.

Direct ``polars.read_parquet`` calls make it impossible to distinguish
decision-time observations from derived reports, durable simulator state, or
hash-bound evidence.  These small readers force that distinction at each call
site and provide one place for schema and integrity checks.

They do not pretend that a derived artifact is point-in-time market data.  A
decision-time snapshot must still use :func:`quant_fund.pit.guarded_read_parquet`.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Collection
from pathlib import Path
from typing import BinaryIO

import polars as pl

type ParquetSource = str | Path | BinaryIO


class ParquetContractError(ValueError):
    """A Parquet artifact does not satisfy its declared read contract."""


def _require_dataset(dataset: str) -> str:
    value = dataset.strip()
    if not value:
        raise ParquetContractError("Parquet dataset name must be non-empty")
    return value


def _validate(
    frame: pl.DataFrame,
    *,
    dataset: str,
    required_columns: Collection[str],
    allow_empty: bool,
) -> pl.DataFrame:
    name = _require_dataset(dataset)
    required = frozenset(required_columns)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ParquetContractError(f"{name}: missing required columns: {missing}")
    if not allow_empty and frame.is_empty():
        raise ParquetContractError(f"{name}: artifact is empty")
    return frame


def _read(
    source: ParquetSource,
    *,
    dataset: str,
    required_columns: Collection[str],
    allow_empty: bool,
) -> pl.DataFrame:
    return _validate(
        pl.read_parquet(source),
        dataset=dataset,
        required_columns=required_columns,
        allow_empty=allow_empty,
    )


def read_market_parquet(
    source: ParquetSource,
    *,
    dataset: str,
    required_columns: Collection[str] = ("event_time",),
    allow_empty: bool = False,
) -> pl.DataFrame:
    """Read a historical market panel with an explicit event-time contract.

    This validates the panel boundary but does not choose a decision-time
    snapshot. Callers making a decision must subsequently use a PIT selector.
    """
    if "event_time" not in required_columns:
        raise ParquetContractError(f"{dataset}: market panels must require event_time")
    return _read(
        source,
        dataset=dataset,
        required_columns=required_columns,
        allow_empty=allow_empty,
    )


def read_vendor_parquet(
    source: ParquetSource,
    *,
    dataset: str,
    allow_empty: bool = False,
) -> pl.DataFrame:
    """Read untrusted vendor-shaped input before adapter normalization."""
    return _read(
        source,
        dataset=dataset,
        required_columns=(),
        allow_empty=allow_empty,
    )


def read_derived_parquet(
    source: ParquetSource,
    *,
    dataset: str,
    required_columns: Collection[str],
    allow_empty: bool = False,
) -> pl.DataFrame:
    """Read a generated feature, label, report, or backtest artifact."""
    return _read(
        source,
        dataset=dataset,
        required_columns=required_columns,
        allow_empty=allow_empty,
    )


def read_state_parquet(
    source: ParquetSource,
    *,
    dataset: str,
    required_columns: Collection[str],
    allow_empty: bool = True,
) -> pl.DataFrame:
    """Read durable simulator or paper-ledger state."""
    return _read(
        source,
        dataset=dataset,
        required_columns=required_columns,
        allow_empty=allow_empty,
    )


def read_evidence_parquet(
    payload: bytes,
    *,
    dataset: str,
    expected_sha256: str,
    required_columns: Collection[str],
    allow_empty: bool = False,
) -> pl.DataFrame:
    """Read immutable evidence only after checking its expected digest."""
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ParquetContractError(
            f"{_require_dataset(dataset)}: sha256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return _read(
        io.BytesIO(payload),
        dataset=dataset,
        required_columns=required_columns,
        allow_empty=allow_empty,
    )


def parquet_schema(path: str | Path, *, dataset: str) -> pl.Schema:
    """Inspect a Parquet schema without materializing its rows."""
    _require_dataset(dataset)
    return pl.scan_parquet(path).collect_schema()


def parquet_row_count(path: str | Path, *, dataset: str) -> int:
    """Count Parquet rows through the same declared metadata boundary."""
    _require_dataset(dataset)
    return int(pl.scan_parquet(path).select(pl.len()).collect().item())
