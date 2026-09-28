"""Evidence chain-of-custody: audit the committed receipts set as a whole.

``verify_receipt_file`` proves a single receipt's integrity. This module adds
the checks a single-file verifier cannot express:

- **filename↔digest binding** — a receipt named ``*_<hex16>.json`` claims its
  ``receipt_sha256`` prefix in the filename; a mismatch means the file was
  renamed or re-saved after sealing (custody break).
- **duplicate seals** — two files carrying the same ``receipt_sha256`` is
  duplicated evidence (or a copy someone forgot they made).
- **coverage accounting** — legacy receipts written before the
  ``receipt_sha256`` envelope are unsealed: parseable artifacts, but their
  integrity cannot be re-verified. They are counted and reported as
  ``n_unsealed`` rather than retroactively failed — they were committed under
  the convention of their era. A *sealed* receipt that fails verification is
  a hard finding.
- **evidence-index freshness** — ``docs/evidence/index.md`` is committed and
  derived from ``receipts/``; regenerating and byte-comparing catches a stale
  or hand-edited index.

The audit emits a sealed ``evidence_audit`` receipt so the audit itself is
part of the evidence base it describes.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from quant_fund.research.fleet_eval import _atomic_write_text
from quant_fund.research.receipt_v2 import (
    build_receipt_v2,
    seal_receipt,
    verify_receipt_file,
)

EVIDENCE_AUDIT_SCHEMA = "evidence_audit.v1"
EVIDENCE_AUDIT_KIND = "evidence_audit"

_DIGEST_SUFFIX = re.compile(r"_([0-9a-f]{16})\.json$")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _filename_digest_status(path: Path, seal: object) -> bool | None:
    """True/False when the filename claims a digest prefix; None when the
    filename does not use the ``*_<hex16>`` convention or no seal exists."""
    match = _DIGEST_SUFFIX.search(path.name)
    if match is None or not isinstance(seal, str):
        return None
    return seal[:16] == match.group(1)


def audit_receipts_dir(receipts_dir: Path | str) -> list[dict[str, Any]]:
    """Verify every ``*.json`` in ``receipts_dir``. Never raises per-file:
    an unreadable or unparseable file is a row with ``valid=False``."""
    directory = Path(receipts_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        result = verify_receipt_file(path)
        try:
            seal = json.loads(path.read_text()).get("receipt_sha256")
        except (OSError, UnicodeError, json.JSONDecodeError):
            seal = None
        sealed = isinstance(seal, str) and len(seal) == 64
        rows.append(
            {
                "file": path.name,
                "file_sha256": _file_sha256(path),
                "schema": result["schema"],
                "kind": result["kind"],
                "verdict": result["verdict"],
                "sealed": sealed,
                "valid": bool(result["valid"]),
                "digest_convention": result["digest_convention"],
                "filename_digest_ok": _filename_digest_status(path, seal),
                "errors": list(result["errors"]),
            }
        )
    return rows


def evidence_audit_findings(
    rows: list[dict[str, Any]], *, index_fresh: bool | None = None
) -> list[str]:
    """Set-level findings: hard violations only (unsealed files are reported
    in the payload, not failed)."""
    findings: list[str] = []
    if not rows:
        findings.append("no_receipts_found")
    for row in rows:
        if row["sealed"] and not row["valid"]:
            findings.append(f"{row['file']}:sealed_receipt_invalid")
        if any(e.startswith("receipt_unreadable") for e in row["errors"]):
            findings.append(f"{row['file']}:unparseable")
        if row["filename_digest_ok"] is False:
            findings.append(f"{row['file']}:filename_digest_mismatch")
    if index_fresh is False:
        findings.append("evidence_index_stale")
    return findings


def check_evidence_index_fresh(root: Path) -> bool | None:
    """Regenerate ``docs/evidence/index.md`` to a temp file and byte-compare.
    Returns None when the builder cannot run (missing script, nonzero exit)."""
    script = root / "scripts" / "build_evidence_report.py"
    index_path = root / "docs" / "evidence" / "index.md"
    if not script.is_file() or not index_path.is_file():
        return None
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as tmp:
        tmp_path = Path(tmp.name)
    try:
        proc = subprocess.run(
            [sys.executable, str(script), "--root", str(root), "--out", str(tmp_path)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return None
        regenerated = tmp_path.read_bytes()
        committed = index_path.read_bytes()
        return regenerated == committed
    finally:
        tmp_path.unlink(missing_ok=True)


def run_evidence_audit(
    receipts_dir: Path | str = Path("receipts"),
    *,
    check_index: bool = True,
    root: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Audit the receipts directory and return (rows, sealed receipt.v2)."""
    directory = Path(receipts_dir)
    if not directory.is_dir():
        raise ValueError(f"receipts directory not found: {directory}")
    rows = audit_receipts_dir(directory)

    # duplicate-seal detection needs the seal itself — re-verify cheaply via
    # the JSON bodies already on disk (verify_receipt_file does not echo it).
    seal_by_file: dict[str, str] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            seal = json.loads(path.read_text()).get("receipt_sha256")
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(seal, str) and len(seal) == 64:
            seal_by_file[path.name] = seal
    dup: dict[str, list[str]] = {}
    for name, seal in seal_by_file.items():
        dup.setdefault(seal, []).append(name)
    duplicate_seals = {seal: sorted(names) for seal, names in dup.items() if len(names) > 1}

    index_fresh = (
        check_evidence_index_fresh(Path(root)) if check_index and root is not None else None
    )

    findings = evidence_audit_findings(rows, index_fresh=index_fresh)
    findings.extend(
        f"duplicate_seal:{seal[:16]}:{','.join(names)}"
        for seal, names in sorted(duplicate_seals.items())
    )

    n_sealed = sum(1 for r in rows if r["sealed"])
    n_valid = sum(1 for r in rows if r["valid"])
    payload: dict[str, Any] = {
        "schema": EVIDENCE_AUDIT_SCHEMA,
        "receipts_dir": str(directory),
        "check_index": bool(check_index),
        "index_fresh": index_fresh,
        "n_files": len(rows),
        "n_sealed": n_sealed,
        "n_unsealed": len(rows) - n_sealed,
        "n_valid": n_valid,
        "n_invalid_sealed": n_sealed - sum(1 for r in rows if r["sealed"] and r["valid"]),
        "duplicate_seals": duplicate_seals,
        "findings": findings,
        "files": rows,
        "live_pnl_claim": False,
    }
    verdict = "pass" if not findings else "fail"
    dataset = {
        "receipts_dir": str(directory),
        "file_digests": {r["file"]: r["file_sha256"] for r in rows},
    }
    params = {"check_index": bool(check_index), "root": str(root) if root else None}
    receipt = build_receipt_v2(
        kind=EVIDENCE_AUDIT_KIND,
        data_label="META",
        dataset=dataset,
        params=params,
        code_files=(Path(__file__),),
        verdict=verdict,
        payload=payload,
    )
    return rows, seal_receipt(receipt)


def evidence_audit_contract_errors(receipt: object) -> list[str]:
    """Fail-closed contract for ``evidence_audit`` receipts."""
    if not isinstance(receipt, dict):
        return ["receipt_not_object"]
    errors: list[str] = []
    if receipt.get("kind") != EVIDENCE_AUDIT_KIND:
        errors.append("kind_mismatch")
    if receipt.get("data_label") != "META":
        errors.append("data_label_must_be_META")
    payload = receipt.get("payload")
    if not isinstance(payload, dict):
        errors.append("payload_missing")
    else:
        if payload.get("schema") != EVIDENCE_AUDIT_SCHEMA:
            errors.append("payload_schema_mismatch")
        if payload.get("live_pnl_claim") is not False:
            errors.append("payload_live_pnl_claim_not_false")
        files = payload.get("files")
        if not isinstance(files, list):
            errors.append("payload_files_missing")
        elif payload.get("n_files") != len(files):
            errors.append("payload_n_files_mismatch")
    return errors


def evidence_audit_consistency_errors(body: object) -> list[str]:
    """Re-derive the audit's counts and verdict from the sealed file rows —
    catches a tampered summary without re-reading the receipts directory."""
    if not isinstance(body, dict):
        return []
    payload = body.get("payload")
    if not isinstance(payload, dict):
        return []
    files = payload.get("files")
    if not isinstance(files, list):
        return []
    errors: list[str] = []
    n_sealed = sum(1 for r in files if isinstance(r, dict) and r.get("sealed"))
    n_valid = sum(1 for r in files if isinstance(r, dict) and r.get("valid"))
    checks = {
        "n_files": len(files),
        "n_sealed": n_sealed,
        "n_unsealed": len(files) - n_sealed,
        "n_valid": n_valid,
        "n_invalid_sealed": n_sealed
        - sum(1 for r in files if isinstance(r, dict) and r.get("sealed") and r.get("valid")),
    }
    for key, want in checks.items():
        if payload.get(key) != want:
            errors.append(f"count_mismatch:{key}")
    findings = payload.get("findings")
    if isinstance(findings, list):
        expected = [
            f"{r['file']}:sealed_receipt_invalid"
            for r in files
            if isinstance(r, dict) and r.get("sealed") and not r.get("valid")
        ]
        embedded = {str(f) for f in findings}
        for want_finding in expected:
            if want_finding not in embedded:
                errors.append(f"finding_missing:{want_finding}")
        want_verdict = "pass" if not findings else "fail"
        if body.get("verdict") != want_verdict:
            errors.append("verdict_findings_inconsistent")
    dup = payload.get("duplicate_seals")
    if isinstance(dup, dict):
        names_in_files = {r.get("file") for r in files if isinstance(r, dict) and r.get("file")}
        for seal, names in dup.items():
            if not isinstance(seal, str) or len(seal) != 64:
                errors.append("duplicate_seal_key_not_hex64")
            if not isinstance(names, list) or len(names) < 2:
                errors.append(f"duplicate_seal_group_too_small:{str(seal)[:16]}")
            elif not set(names).issubset(names_in_files):
                errors.append(f"duplicate_seal_unknown_file:{str(seal)[:16]}")
    return errors


def write_evidence_audit_receipt(
    receipt: dict[str, Any], out_dir: Path | str = Path("receipts")
) -> Path:
    """Write the sealed audit receipt as ``evidence_audit_<digest16>.json``."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    errors = evidence_audit_contract_errors(receipt)
    if errors:
        raise ValueError(f"evidence_audit contract: {errors}")
    digest = receipt.get("receipt_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("receipt must be sealed before writing")
    path = directory / f"evidence_audit_{digest[:16]}.json"
    _atomic_write_text(path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return path


def format_evidence_audit_table(rows: list[dict[str, Any]]) -> str:
    """Render the audit rows as a fixed-width table for CLI output."""
    header = f"{'file':<52} {'sealed':>6} {'valid':>5} {'name-digest':>11} errors"
    lines = [header, "-" * len(header)]
    for row in rows:
        fd = row["filename_digest_ok"]
        fd_str = "-" if fd is None else ("ok" if fd else "MISMATCH")
        errs = ";".join(row["errors"]) if row["errors"] else "-"
        lines.append(
            f"{row['file']:<52} {str(row['sealed']):>6} {str(row['valid']):>5} "
            f"{fd_str:>11} {errs[:60]}"
        )
    return "\n".join(lines)
