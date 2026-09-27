"""Optional, additive receipt extension for explainability reports.

The sealed research receipt is immutable: ``verify_research_artifact``
requires the mutable ``latest.json`` to equal the immutable ``runs/<run_id>.json``
byte-for-byte (``immutable_receipt_mismatch``) and binds it via
``artifacts.immutable_json_sha256``. Retrofitting a field into an existing
receipt would break verification — so the extension is a **sidecar file**
written next to the receipt, never inside it:

    <receipt_stem>.explainability.json

The sidecar binds the parent receipt by content hash (``receipt.sha256``) plus
``run_id`` when the receipt exposes one, and lists each report artifact with
its own sha256. Sealed receipts and every existing verify path are
untouched — ``verify_research_artifact`` neither knows nor needs to know the
sidecar exists. ``verify_explainability_sidecar`` is an additional, separate
check that the binding still matches the current bytes on disk.

For receipts minted in the future, ``explainability_artifact_entry`` returns
an optional dict that a receipt writer may embed under
``artifacts.explainability`` *at write time* (the immutable and mutable copies
must both carry it). Unknown artifact keys are ignored by the verifier, so
this stays backward-compatible — but retrofitting it into an already-sealed
receipt remains forbidden.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_fund.research.explainability.report import (
    ExplainabilityReport,
    _atomic_write,
    write_report,
)
from quant_fund.utils.hashing import hash_bytes, hash_file

EXPLAINABILITY_SIDECAR_SCHEMA = "explainability_attachment.v1"
EXPLAINABILITY_ARTIFACT_SCHEMA = "explainability_artifacts.v1"

_KIND_BY_SUFFIX = {
    ".md": "markdown",
    ".html": "html",
    ".json": "json",
    ".png": "image",
    ".svg": "image",
}


def explainability_sidecar_path(receipt_path: Path | str) -> Path:
    """Sidecar location for a receipt file: ``<stem>.explainability.json`` beside it."""
    receipt = Path(receipt_path)
    return receipt.with_name(f"{receipt.stem}.explainability.json")


def _receipt_identity(receipt_path: Path) -> dict[str, Any]:
    """Best-effort identity fields parsed from the receipt payload."""
    identity: dict[str, Any] = {}
    try:
        payload = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError):
        return identity
    if not isinstance(payload, dict):
        return identity
    provenance = payload.get("provenance")
    if isinstance(provenance, dict) and isinstance(provenance.get("run_id"), str):
        identity["run_id"] = provenance["run_id"]
    if isinstance(payload.get("schema_version"), int):
        identity["receipt_schema_version"] = payload["schema_version"]
    if isinstance(payload.get("synthetic"), bool):
        identity["synthetic"] = payload["synthetic"]
    return identity


def _report_entries(
    report_paths: list[Path], receipt_dir: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    entries: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in report_paths:
        resolved = Path(path).resolve()
        if not resolved.is_file():
            errors.append(f"report_missing:{path}")
            continue
        try:
            relative = resolved.relative_to(receipt_dir)
            location = relative.as_posix()
            inside = True
        except ValueError:
            location = str(resolved)
            inside = False
        entries.append(
            {
                "path": location,
                "inside_receipt_root": inside,
                "sha256": hash_file(resolved),
                "bytes": resolved.stat().st_size,
                "kind": _KIND_BY_SUFFIX.get(resolved.suffix.lower(), "file"),
            }
        )
    return entries, errors


def attach_explainability_sidecar(
    receipt_path: Path | str,
    report_paths: list[Path | str] | Mapping[str, Path | str] | Path | str,
    *,
    generated_at: str | None = None,
) -> Path:
    """Write ``<receipt>.explainability.json`` binding reports to a sealed receipt.

    ``report_paths`` may be a directory (its ``explainability.*`` files are
    collected), a list of files, or a ``{kind: path}`` mapping. The receipt
    file itself is opened read-only and never modified.
    """
    receipt = Path(receipt_path)
    if not receipt.is_file():
        raise FileNotFoundError(f"receipt not found: {receipt}")
    receipt_dir = receipt.resolve().parent

    if isinstance(report_paths, (str, Path)):
        candidate = Path(report_paths)
        if candidate.is_dir():
            paths = sorted(candidate.glob("explainability.*"))
            paths = [p for p in paths if p.is_file() and not p.name.endswith(".sha256")]
        else:
            paths = [candidate]
    elif isinstance(report_paths, Mapping):
        paths = [Path(v) for v in report_paths.values()]
    else:
        paths = [Path(p) for p in report_paths]
    if not paths:
        raise ValueError("no explainability report files to attach")

    entries, errors = _report_entries(paths, receipt_dir)
    if errors:
        raise FileNotFoundError("; ".join(errors))

    sidecar = {
        "schema": EXPLAINABILITY_SIDECAR_SCHEMA,
        "generated_at": generated_at or datetime.now(UTC).isoformat(),
        "claim": "research_only",
        "receipt": {
            "file": receipt.name,
            "sha256": hash_file(receipt),
            **_receipt_identity(receipt),
        },
        "reports": entries,
    }
    sidecar["sidecar_sha256_payload"] = hash_bytes(
        json.dumps(
            {k: v for k, v in sidecar.items() if k != "sidecar_sha256_payload"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    destination = explainability_sidecar_path(receipt)
    _atomic_write(
        destination,
        (json.dumps(sidecar, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return destination


def attach_explainability_report(
    receipt_path: Path | str,
    report: ExplainabilityReport,
    *,
    report_dir: Path | str | None = None,
    generated_at: str | None = None,
) -> dict[str, Path]:
    """Write report files under ``<receipt_dir>/explainability/`` and attach the sidecar."""
    receipt = Path(receipt_path)
    out_dir = Path(report_dir) if report_dir is not None else receipt.parent / "explainability"
    paths = write_report(report, out_dir)
    sidecar = attach_explainability_sidecar(receipt, paths, generated_at=generated_at)
    return {**paths, "sidecar": sidecar}


def _resolve_report_path(entry_path: str, receipt_dir: Path) -> Path:
    candidate = Path(entry_path)
    return candidate if candidate.is_absolute() else receipt_dir / candidate


def verify_explainability_sidecar(
    receipt_path: Path | str,
    sidecar_path: Path | str | None = None,
) -> dict[str, Any]:
    """Additive check: the sidecar binding still matches the bytes on disk.

    Returns ``{"valid": bool, "errors": [...], "reports_checked": int}``.
    This never replaces ``verify_research_artifact`` — a valid receipt can be
    unexplainable, and an explainability sidecar says nothing about the
    receipt's own validity beyond hash binding.
    """
    errors: list[str] = []
    receipt = Path(receipt_path)
    sidecar = (
        Path(sidecar_path) if sidecar_path is not None else explainability_sidecar_path(receipt)
    )
    if not receipt.is_file():
        return {"valid": False, "errors": ["receipt_missing"], "reports_checked": 0}
    if not sidecar.is_file():
        return {"valid": False, "errors": ["sidecar_missing"], "reports_checked": 0}
    try:
        payload = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "errors": [f"sidecar_unreadable:{exc}"], "reports_checked": 0}
    if not isinstance(payload, dict) or payload.get("schema") != EXPLAINABILITY_SIDECAR_SCHEMA:
        errors.append("sidecar_schema_mismatch")
        payload = {}

    raw_receipt = payload.get("receipt")
    receipt_info = raw_receipt if isinstance(raw_receipt, dict) else {}
    if receipt_info.get("file") != receipt.name:
        errors.append("sidecar_receipt_name_mismatch")
    expected_receipt_hash = receipt_info.get("sha256")
    if not isinstance(expected_receipt_hash, str) or hash_file(receipt) != expected_receipt_hash:
        errors.append("sidecar_receipt_hash_mismatch")

    identity = _receipt_identity(receipt)
    if "run_id" in identity and receipt_info.get("run_id") != identity["run_id"]:
        errors.append("sidecar_run_id_mismatch")

    reports = payload.get("reports")
    checked = 0
    if not isinstance(reports, list) or not reports:
        errors.append("sidecar_reports_missing")
    else:
        receipt_dir = receipt.resolve().parent
        for index, entry in enumerate(reports):
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                errors.append(f"sidecar_report_entry_invalid:{index}")
                continue
            resolved = _resolve_report_path(entry["path"], receipt_dir)
            if not resolved.is_file():
                errors.append(f"sidecar_report_missing:{entry['path']}")
                continue
            expected = entry.get("sha256")
            if not isinstance(expected, str) or hash_file(resolved) != expected:
                errors.append(f"sidecar_report_hash_mismatch:{entry['path']}")
                continue
            checked += 1
    return {"valid": not errors, "errors": errors, "reports_checked": checked}


def explainability_artifact_entry(
    report_dir: Path | str,
    receipt_dir: Path | str,
) -> dict[str, Any]:
    """Optional ``artifacts.explainability`` entry for receipts minted *now*.

    Returns a dict of relative report paths + sha256 digests suitable for
    embedding in a NEW receipt's ``artifacts`` mapping before sealing (both
    the immutable and mutable copies must contain the entry, so it can only
    be added at write time — never retrofitted onto a sealed receipt). The
    existing verifier ignores unknown artifact keys, keeping the entry
    additive and backward-compatible.
    """
    reports = Path(report_dir)
    root = Path(receipt_dir).resolve()
    if not reports.is_dir():
        raise FileNotFoundError(f"report dir not found: {reports}")
    files = sorted(p for p in reports.glob("explainability.*") if p.is_file())
    entry: dict[str, Any] = {"schema": EXPLAINABILITY_ARTIFACT_SCHEMA}
    for path in files:
        if path.name.endswith(".sha256"):
            continue
        kind = _KIND_BY_SUFFIX.get(path.suffix.lower(), "file")
        try:
            location = path.resolve().relative_to(root).as_posix()
        except ValueError:
            location = str(path.resolve())
        entry[kind] = location
        entry[f"{kind}_sha256"] = hash_file(path)
    return entry
