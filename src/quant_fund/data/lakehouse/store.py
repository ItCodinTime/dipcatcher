"""Immutable snapshots over a content-addressed Parquet object store."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

import polars as pl

from quant_fund.schemas.errors import DataContractError
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import ContentAddress, content_address

SNAPSHOT_SCHEMA = "dipcatcher.lake.snapshot.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"^[A-Za-z0-9._-]+$")
_BAR_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True)
class FileEntry:
    """One object in a snapshot. Paths are relative to the lake root."""

    logical_path: str
    object_path: str
    content_sha256: str
    stored_sha256: str
    stored_size: int
    declared_size: int
    kind: str
    role: str
    source: str | None
    symbol: str | None
    date: str | None
    byte_identical_to_source: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_path": self.logical_path,
            "object_path": self.object_path,
            "content_sha256": self.content_sha256,
            "stored_sha256": self.stored_sha256,
            "stored_size": self.stored_size,
            "declared_size": self.declared_size,
            "kind": self.kind,
            "role": self.role,
            "source": self.source,
            "symbol": self.symbol,
            "date": self.date,
            "byte_identical_to_source": self.byte_identical_to_source,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FileEntry:
        try:
            entry = cls(
                logical_path=_require_relpath(raw["logical_path"], field="logical_path"),
                object_path=_require_relpath(raw["object_path"], field="object_path"),
                content_sha256=_require_sha(raw["content_sha256"], field="content_sha256"),
                stored_sha256=_require_sha(raw["stored_sha256"], field="stored_sha256"),
                stored_size=_require_size(raw["stored_size"], field="stored_size"),
                declared_size=_require_size(raw["declared_size"], field="declared_size"),
                kind=_require_kind(raw["kind"]),
                role=_require_token(raw["role"], field="role"),
                source=_optional_token(raw.get("source"), field="source"),
                symbol=_optional_token(raw.get("symbol"), field="symbol"),
                date=_optional_date(raw.get("date")),
                byte_identical_to_source=raw["byte_identical_to_source"],
            )
        except (KeyError, TypeError) as exc:
            raise DataContractError(f"snapshot file entry is malformed: {exc}") from exc
        if not isinstance(entry.byte_identical_to_source, bool):
            raise DataContractError("byte_identical_to_source must be a bool")
        return entry


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str
    dataset: str
    files: tuple[FileEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SNAPSHOT_SCHEMA,
            "snapshot_id": self.snapshot_id,
            "dataset": self.dataset,
            "files": [entry.to_dict() for entry in self.files],
        }


def snapshot_id_for(dataset: str, files: tuple[FileEntry, ...] | list[FileEntry]) -> str:
    """Content id of a manifest. The id itself is not part of the preimage."""
    ordered = tuple(sorted(files, key=lambda entry: entry.logical_path))
    payload = {
        "schema": SNAPSHOT_SCHEMA,
        "dataset": dataset,
        "files": [entry.to_dict() for entry in ordered],
    }
    return hash_bytes(canonical_json_bytes(payload))


def object_relpath(stored_sha256: str) -> str:
    digest = _require_sha(stored_sha256, field="stored_sha256")
    return f"objects/sha256/{digest[:2]}/{digest}"


def commit_snapshot(root: Path, dataset: str, files: list[FileEntry]) -> Snapshot:
    """Write an immutable snapshot. The same manifest is idempotent."""
    label = _require_dataset(dataset)
    ordered = tuple(sorted(files, key=lambda entry: entry.logical_path))
    logical = [entry.logical_path for entry in ordered]
    if len(logical) != len(set(logical)):
        raise DataContractError("snapshot logical paths must be unique")
    snapshot_id = snapshot_id_for(label, ordered)
    snapshot = Snapshot(snapshot_id=snapshot_id, dataset=label, files=ordered)
    path = Path(root) / "snapshots" / f"{snapshot_id}.json"
    body = canonical_json_bytes(snapshot.to_dict())
    if path.is_file():
        existing = path.read_bytes()
        if existing != body:
            raise DataContractError(f"snapshot {snapshot_id} already exists with different bytes")
        return snapshot
    _atomic_write(path, body)
    return snapshot


def load_snapshot(root: Path, snapshot_id: str) -> Snapshot:
    digest = _require_sha(snapshot_id, field="snapshot_id")
    path = Path(root) / "snapshots" / f"{digest}.json"
    if not path.is_file():
        raise DataContractError(f"snapshot not found: {digest}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataContractError(f"snapshot {digest} is malformed") from exc
    if not isinstance(raw, dict) or raw.get("schema") != SNAPSHOT_SCHEMA:
        raise DataContractError(f"snapshot {digest} has an unsupported schema")
    dataset = raw.get("dataset")
    files_raw = raw.get("files")
    if not isinstance(dataset, str) or not isinstance(files_raw, list):
        raise DataContractError(f"snapshot {digest} is malformed")
    files = tuple(FileEntry.from_dict(item) for item in files_raw if isinstance(item, dict))
    if len(files) != len(files_raw):
        raise DataContractError(f"snapshot {digest} has a non-object file entry")
    expected = snapshot_id_for(dataset, files)
    if expected != digest or raw.get("snapshot_id") != digest:
        raise DataContractError(f"snapshot {digest} does not match its manifest bytes")
    return Snapshot(snapshot_id=digest, dataset=dataset, files=files)


def put_file(root: Path, source: Path) -> tuple[Path, ContentAddress]:
    """Copy ``source`` into the object store without rewriting its bytes."""
    address = content_address(source)
    destination = Path(root) / object_relpath(address.stored_sha256)
    if destination.is_file():
        existing = content_address(destination)
        if existing.stored_sha256 != address.stored_sha256:
            raise DataContractError(f"object {address.stored_sha256} exists with different bytes")
        return destination, address
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        _copy_exact(source, temporary)
        copied = content_address(temporary)
        if (
            copied.stored_sha256 != address.stored_sha256
            or copied.stored_size != address.stored_size
        ):
            raise DataContractError("copied object bytes do not match the source")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination, address


def write_partitioned_bars(root: Path, frame: pl.DataFrame, *, dataset: str) -> Snapshot:
    """Write hive-style ``source/symbol/date`` Parquet parts and snapshot them.

    Each part is stored once under its content hash. This path creates new
    bytes; byte-preserving import of an existing file is :func:`import_files`.
    """
    symbol_col = _symbol_column(frame)
    missing = [
        name for name in ("source", "event_time", *_BAR_COLUMNS) if name not in frame.columns
    ]
    if missing:
        raise DataContractError(f"partitioned bars missing columns: {missing}")
    if frame.is_empty():
        return commit_snapshot(root, dataset, [])
    _require_utc_datetime(frame, "event_time")
    dated = frame.with_columns(pl.col("event_time").dt.date().alias("_lake_date"))
    grouped = cast(
        dict[tuple[object, ...], pl.DataFrame],
        dated.partition_by(["source", symbol_col, "_lake_date"], as_dict=True),
    )
    entries: list[FileEntry] = []
    for key, part in grouped.items():
        if not isinstance(key, tuple) or len(key) != 3:
            raise DataContractError(f"unexpected partition key: {key!r}")
        source, symbol, day = key
        source_token = _require_token(source, field="source")
        symbol_token = _require_token(symbol, field="symbol")
        day_text = _date_text(day)
        body = part.drop("_lake_date")
        logical = f"partitions/source={source_token}/symbol={symbol_token}/date={day_text}"
        entry = _write_part(
            root,
            body,
            logical_path=f"{logical}/part.parquet",
            role="bars",
            source=source_token,
            symbol=symbol_token,
            day=day_text,
        )
        entries.append(entry)
    return commit_snapshot(root, dataset, entries)


def file_entry_for_object(
    *,
    logical_path: str,
    object_path: str,
    address: ContentAddress,
    role: str,
    source: str | None,
    symbol: str | None,
    day: str | None,
    byte_identical_to_source: bool,
) -> FileEntry:
    return FileEntry(
        logical_path=_require_relpath(logical_path, field="logical_path"),
        object_path=_require_relpath(object_path, field="object_path"),
        content_sha256=address.content_sha256,
        stored_sha256=address.stored_sha256,
        stored_size=address.stored_size,
        declared_size=address.declared_size,
        kind=address.kind,
        role=_require_token(role, field="role"),
        source=_optional_token(source, field="source"),
        symbol=_optional_token(symbol, field="symbol"),
        date=_optional_date(day),
        byte_identical_to_source=byte_identical_to_source,
    )


def _write_part(
    root: Path,
    frame: pl.DataFrame,
    *,
    logical_path: str,
    role: str,
    source: str,
    symbol: str,
    day: str,
) -> FileEntry:
    lake = Path(root)
    staging = lake / "tmp"
    staging.mkdir(parents=True, exist_ok=True)
    temporary = staging / f"{os.getpid()}-{hash_bytes(logical_path.encode())}.parquet"
    try:
        frame.write_parquet(temporary)
        destination, address = put_file(lake, temporary)
    finally:
        temporary.unlink(missing_ok=True)
    # The logical name ends in a stable suffix; the object name is the hash.
    hashed_logical = str(Path(logical_path).with_name(f"{address.stored_sha256}.parquet"))
    return file_entry_for_object(
        logical_path=hashed_logical,
        object_path=destination.relative_to(lake).as_posix(),
        address=address,
        role=role,
        source=source,
        symbol=symbol,
        day=day,
        byte_identical_to_source=False,
    )


def _copy_exact(source: Path, destination: Path) -> None:
    with source.open("rb") as incoming, destination.open("wb") as outgoing:
        while True:
            chunk = incoming.read(1 << 20)
            if not chunk:
                break
            outgoing.write(chunk)
        outgoing.flush()
        os.fsync(outgoing.fileno())


def _atomic_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _symbol_column(frame: pl.DataFrame) -> str:
    if "symbol" in frame.columns:
        return "symbol"
    if "security_id" in frame.columns:
        return "security_id"
    raise DataContractError("partitioned bars need symbol or security_id")


def _require_utc_datetime(frame: pl.DataFrame, column: str) -> None:
    dtype = frame.schema.get(column)
    if not isinstance(dtype, pl.Datetime) or dtype.time_zone != "UTC":
        raise DataContractError(f"{column} must be a UTC datetime")


def _date_text(value: object) -> str:
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value.isoformat()
    raise DataContractError(f"partition date is not a date: {value!r}")


def _require_dataset(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\n" in value or "\x00" in value:
        raise DataContractError("dataset name must be a non-empty single line")
    return value


def _require_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DataContractError(f"{field} must be a lowercase sha256")
    return value


def _require_size(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataContractError(f"{field} must be a non-negative integer")
    return value


def _require_kind(value: object) -> str:
    if value not in {"blob", "lfs_pointer"}:
        raise DataContractError("file kind must be blob or lfs_pointer")
    return str(value)


def _require_token(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        raise DataContractError(f"unsafe {field} for a lake path: {value!r}")
    return value


def _optional_token(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _require_token(value, field=field)


def _optional_date(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 10:
        raise DataContractError("partition date must be YYYY-MM-DD")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise DataContractError("partition date must be YYYY-MM-DD") from exc
    return value


def _require_relpath(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value.startswith(("/", "\\")):
        raise DataContractError(f"{field} must be a relative path")
    parts = Path(value).parts
    if ".." in parts or any(part in {"", ".", ".."} or "\\" in part for part in parts):
        raise DataContractError(f"{field} must stay inside the lake")
    return Path(value).as_posix()
