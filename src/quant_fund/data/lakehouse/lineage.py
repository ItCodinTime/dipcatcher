"""Lineage records: input snapshots, code hash, and parameters."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from quant_fund.data.lakehouse.store import load_snapshot
from quant_fund.schemas.errors import DataContractError
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import content_address

LINEAGE_SCHEMA = "dipcatcher.lake.lineage.v1"
INDEX_SCHEMA = "dipcatcher.lake.lineage-index.v1"


@dataclass(frozen=True)
class LineageInput:
    dataset: str
    snapshot_id: str

    def to_dict(self) -> dict[str, str]:
        return {"dataset": self.dataset, "snapshot_id": self.snapshot_id}


@dataclass(frozen=True)
class LineageRecord:
    record_id: str
    dataset: str
    snapshot_id: str
    inputs: tuple[LineageInput, ...]
    code_hash: str
    code_path: str | None
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LINEAGE_SCHEMA,
            "record_id": self.record_id,
            "dataset": self.dataset,
            "snapshot_id": self.snapshot_id,
            "inputs": [item.to_dict() for item in self.inputs],
            "code_hash": self.code_hash,
            "code_path": self.code_path,
            "parameters": self.parameters,
        }


@dataclass(frozen=True)
class DriftReport:
    ok: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": list(self.errors)}


def record_lineage(
    root: Path,
    *,
    dataset: str,
    snapshot_id: str,
    inputs: Sequence[LineageInput] = (),
    code: str | Path | Callable[..., Any],
    parameters: dict[str, Any] | None = None,
    code_path: str | None = None,
) -> LineageRecord:
    """Append an immutable lineage record and point the dataset index at it."""
    load_snapshot(root, snapshot_id)
    for item in inputs:
        load_snapshot(root, item.snapshot_id)
    digest, bound_path = _code_identity(code, code_path)
    params = parameters or {}
    canonical_json_bytes(params)
    body = {
        "schema": LINEAGE_SCHEMA,
        "dataset": dataset,
        "snapshot_id": snapshot_id,
        "inputs": [item.to_dict() for item in sorted(inputs, key=lambda item: item.dataset)],
        "code_hash": digest,
        "code_path": bound_path,
        "parameters": params,
    }
    record_id = hash_bytes(canonical_json_bytes(body))
    record = LineageRecord(
        record_id=record_id,
        dataset=dataset,
        snapshot_id=snapshot_id,
        inputs=tuple(sorted(inputs, key=lambda item: item.dataset)),
        code_hash=digest,
        code_path=bound_path,
        parameters=params,
    )
    path = Path(root) / "lineage" / "records" / f"{record_id}.json"
    payload = canonical_json_bytes(record.to_dict())
    if path.is_file():
        if path.read_bytes() != payload:
            raise DataContractError(f"lineage record {record_id} exists with different bytes")
    else:
        _write(path, payload)
    _point_index(root, dataset, record_id)
    return record


def format_dag(root: Path, dataset: str) -> str:
    """Text DAG for ``lineage show``. Input snapshots are the pinned ids."""
    lines: list[str] = []
    _render(root, dataset, "", set(), lines, pinned_snapshot=None)
    if not lines:
        raise DataContractError(f"no lineage for dataset {dataset}")
    return "\n".join(lines) + "\n"


def verify_lineage(root: Path, dataset: str | None = None) -> DriftReport:
    """Recompute snapshot, object, and code hashes. Drift is a failed report."""
    errors: list[str] = []
    names = [dataset] if dataset is not None else sorted(_index(root))
    if dataset is not None and dataset not in _index(root):
        errors.append(f"lineage_missing:{dataset}")
    seen: set[tuple[str, str]] = set()
    for name in names:
        _verify_dataset(root, name, errors, seen)
    return DriftReport(ok=not errors, errors=tuple(errors))


def _verify_dataset(
    root: Path,
    dataset: str,
    errors: list[str],
    seen: set[tuple[str, str]],
) -> None:
    index = _index(root)
    record_id = index.get(dataset)
    if record_id is None:
        errors.append(f"lineage_missing:{dataset}")
        return
    try:
        record = _load_record(root, record_id)
    except DataContractError as exc:
        errors.append(f"lineage_record_unreadable:{dataset}:{exc}")
        return
    if record.dataset != dataset:
        errors.append(f"lineage_index_mismatch:{dataset}")
    key = (record.dataset, record.snapshot_id)
    if key in seen:
        errors.append(f"lineage_cycle:{dataset}")
        return
    seen.add(key)
    try:
        snapshot = load_snapshot(root, record.snapshot_id)
    except DataContractError as exc:
        errors.append(f"snapshot_unreadable:{record.snapshot_id}:{exc}")
        snapshot = None
    if snapshot is not None:
        for entry in snapshot.files:
            object_path = Path(root) / entry.object_path
            if not object_path.is_file():
                errors.append(f"object_missing:{entry.object_path}")
                continue
            try:
                address = content_address(object_path)
            except OSError as exc:
                errors.append(f"object_unreadable:{entry.object_path}:{exc}")
                continue
            if address.stored_sha256 != entry.stored_sha256:
                errors.append(f"stored_hash_drift:{entry.object_path}")
            if address.content_sha256 != entry.content_sha256 or address.kind != entry.kind:
                errors.append(f"content_hash_drift:{entry.object_path}")
    if record.code_path is not None:
        code_file = Path(record.code_path)
        if not code_file.is_file():
            errors.append(f"code_missing:{record.code_path}")
        else:
            current = content_address(code_file)
            if current.kind != "blob" or current.content_sha256 != record.code_hash:
                errors.append(f"code_hash_drift:{record.code_path}")
    for item in record.inputs:
        try:
            load_snapshot(root, item.snapshot_id)
        except DataContractError:
            errors.append(f"input_snapshot_missing:{item.dataset}:{item.snapshot_id}")
            continue
        child = _index(root).get(item.dataset)
        if child is None:
            continue
        try:
            child_record = _load_record(root, child)
        except DataContractError:
            errors.append(f"lineage_record_unreadable:{item.dataset}")
            continue
        if child_record.snapshot_id == item.snapshot_id:
            _verify_dataset(root, item.dataset, errors, seen)


def _render(
    root: Path,
    dataset: str,
    indent: str,
    stack: set[str],
    lines: list[str],
    *,
    pinned_snapshot: str | None,
) -> None:
    if dataset in stack:
        lines.append(f"{indent}{dataset} (cycle)")
        return
    index = _index(root)
    record_id = index.get(dataset)
    if record_id is None:
        if pinned_snapshot is None:
            raise DataContractError(f"no lineage for dataset {dataset}")
        lines.append(f"{indent}{dataset}")
        lines.append(f"{indent}  snapshot {pinned_snapshot}")
        lines.append(f"{indent}  (root snapshot)")
        return
    record = _load_record(root, record_id)
    if pinned_snapshot is not None and record.snapshot_id != pinned_snapshot:
        lines.append(f"{indent}{dataset}")
        lines.append(f"{indent}  snapshot {pinned_snapshot}")
        lines.append(f"{indent}  (root snapshot)")
        return
    lines.append(f"{indent}{dataset}")
    lines.append(f"{indent}  snapshot {record.snapshot_id}")
    lines.append(f"{indent}  code {record.code_hash}")
    params = canonical_json_bytes(record.parameters).decode("utf-8")
    lines.append(f"{indent}  parameters {params}")
    for item in record.inputs:
        lines.append(f"{indent}  input")
        _render(
            root,
            item.dataset,
            indent + "    ",
            stack | {dataset},
            lines,
            pinned_snapshot=item.snapshot_id,
        )


def _code_identity(
    code: str | Path | Callable[..., Any],
    code_path: str | None,
) -> tuple[str, str | None]:
    if isinstance(code, Path):
        address = content_address(code)
        if address.kind != "blob":
            raise DataContractError("code hash requires file bytes, not an LFS pointer")
        return address.content_sha256, code.as_posix()
    if callable(code):
        try:
            source = inspect.getsource(code)
        except (OSError, TypeError) as exc:
            raise DataContractError("cannot read callable source for the code hash") from exc
        return hash_bytes(source.encode()), code_path
    if isinstance(code, str):
        return hash_bytes(code.encode()), code_path
    raise DataContractError("code must be a path, callable, or source string")


def _load_record(root: Path, record_id: str) -> LineageRecord:
    path = Path(root) / "lineage" / "records" / f"{record_id}.json"
    if not path.is_file():
        raise DataContractError(f"lineage record not found: {record_id}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != LINEAGE_SCHEMA:
        raise DataContractError(f"lineage record {record_id} is malformed")
    body = {
        "schema": LINEAGE_SCHEMA,
        "dataset": raw.get("dataset"),
        "snapshot_id": raw.get("snapshot_id"),
        "inputs": raw.get("inputs"),
        "code_hash": raw.get("code_hash"),
        "code_path": raw.get("code_path"),
        "parameters": raw.get("parameters"),
    }
    expected = hash_bytes(canonical_json_bytes(body))
    if expected != record_id or raw.get("record_id") != record_id:
        raise DataContractError(f"lineage record {record_id} does not match its bytes")
    inputs_raw = raw.get("inputs")
    if not isinstance(inputs_raw, list):
        raise DataContractError(f"lineage record {record_id} inputs are malformed")
    inputs: list[LineageInput] = []
    for item in inputs_raw:
        if not isinstance(item, dict):
            raise DataContractError(f"lineage record {record_id} inputs are malformed")
        inputs.append(
            LineageInput(dataset=str(item["dataset"]), snapshot_id=str(item["snapshot_id"]))
        )
    parameters = raw.get("parameters")
    if not isinstance(parameters, dict):
        raise DataContractError(f"lineage record {record_id} parameters are malformed")
    code_path = raw.get("code_path")
    if code_path is not None and not isinstance(code_path, str):
        raise DataContractError(f"lineage record {record_id} code_path is malformed")
    return LineageRecord(
        record_id=record_id,
        dataset=str(raw["dataset"]),
        snapshot_id=str(raw["snapshot_id"]),
        inputs=tuple(inputs),
        code_hash=str(raw["code_hash"]),
        code_path=code_path,
        parameters=cast(dict[str, Any], parameters),
    )


def _index(root: Path) -> dict[str, str]:
    path = Path(root) / "lineage" / "index.json"
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != INDEX_SCHEMA:
        raise DataContractError("lineage index is malformed")
    datasets = raw.get("datasets")
    if not isinstance(datasets, dict):
        raise DataContractError("lineage index is malformed")
    return {str(key): str(value) for key, value in datasets.items()}


def _point_index(root: Path, dataset: str, record_id: str) -> None:
    datasets = _index(root)
    datasets[dataset] = record_id
    body = canonical_json_bytes({"schema": INDEX_SCHEMA, "datasets": datasets})
    _write(Path(root) / "lineage" / "index.json", body)


def _write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(body)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
