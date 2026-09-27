"""Import existing data files without changing their bytes."""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from quant_fund.data.lakehouse.store import (
    FileEntry,
    Snapshot,
    commit_snapshot,
    file_entry_for_object,
    put_file,
)
from quant_fund.schemas.errors import DataContractError
from quant_fund.utils.reproducibility import content_address

_OHLC = {"open", "high", "low", "close"}
_TOKEN = re.compile(r"^[A-Za-z0-9._-]+$")


def import_files(
    paths: list[Path] | tuple[Path, ...],
    root: Path,
    *,
    dataset: str,
    source: str | None = None,
    role: str | None = None,
) -> Snapshot:
    """Copy each file into the object store and record an immutable snapshot.

    The source path is hashed before and after the copy. The stored object is
    the same byte string. Partition columns are recorded when a file is a
    single source/symbol/date; multi-day files stay one object.
    """
    if not paths:
        raise DataContractError("import requires at least one file")
    entries: list[FileEntry] = []
    for raw_path in paths:
        path = Path(raw_path)
        before = content_address(path)
        destination, address = put_file(root, path)
        after = content_address(path)
        if after != before:
            raise DataContractError(f"import mutated source bytes: {path}")
        if content_address(destination).stored_sha256 != address.stored_sha256:
            raise DataContractError(f"stored object drifted from source: {path}")
        inferred_role, inferred_source, symbol, day = _describe(path, address.kind)
        entries.append(
            file_entry_for_object(
                logical_path=_logical_import_path(path),
                object_path=destination.relative_to(Path(root)).as_posix(),
                address=address,
                role=role or inferred_role,
                source=source if source is not None else inferred_source,
                symbol=symbol,
                day=day,
                byte_identical_to_source=True,
            )
        )
    return commit_snapshot(root, dataset, entries)


def _logical_import_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        relative = path.name
    if relative.startswith("/") or ".." in Path(relative).parts:
        raise DataContractError(f"refusing import path: {path}")
    return f"imports/{relative}"


def _describe(path: Path, kind: str) -> tuple[str, str | None, str | None, str | None]:
    if kind == "lfs_pointer":
        return "lfs_pointer", None, None, None
    schema_names = set(pl.scan_parquet(path).collect_schema().names())
    role = _infer_role(path, schema_names)
    source = _single_value(path, schema_names, "source")
    symbol: str | None = None
    day: str | None = None
    symbol_col = (
        "symbol" if "symbol" in schema_names else "ticker" if "ticker" in schema_names else None
    )
    time_col = (
        "event_time"
        if "event_time" in schema_names
        else "timestamp"
        if "timestamp" in schema_names
        else None
    )
    if symbol_col is not None and time_col is not None and role == "bars":
        span = (
            pl.scan_parquet(path)
            .select(
                pl.col(symbol_col).cast(pl.Utf8).n_unique().alias("n_symbol"),
                pl.col(time_col).dt.date().n_unique().alias("n_date"),
                pl.col(symbol_col).cast(pl.Utf8).min().alias("symbol"),
                pl.col(time_col).dt.date().min().alias("day"),
            )
            .collect()
        )
        if int(span["n_symbol"][0]) == 1 and int(span["n_date"][0]) == 1:
            symbol = _partition_token(span["symbol"][0])
            day_value = span["day"][0]
            if symbol is not None and hasattr(day_value, "isoformat"):
                day = day_value.isoformat()
            else:
                symbol = None
    return role, source, symbol, day


def _infer_role(path: Path, names: set[str]) -> str:
    stem = path.stem.lower()
    if (
        stem == "universe"
        or {"asof", "security_id"} <= names
        or {"effective_from", "security_id"} <= names
    ):
        return "universe"
    if "action_type" in names:
        return "corporate_actions"
    if names >= _OHLC or {"ticker", "timestamp"} <= names:
        return "bars"
    if {"security_id", "ticker", "valid_from"} <= names:
        return "security_master"
    return "blob"


def _single_value(path: Path, names: set[str], column: str) -> str | None:
    if column not in names:
        return None
    values = (
        pl.scan_parquet(path)
        .select(pl.col(column).cast(pl.Utf8).drop_nulls().unique())
        .collect()
        .get_column(column)
        .to_list()
    )
    if len(values) == 1:
        return _partition_token(values[0])
    return None


def _partition_token(value: object) -> str | None:
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        return None
    return value
