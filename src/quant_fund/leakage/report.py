"""Leakage scan report assembly + serialization (LeakageReport schema)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from quant_fund.leakage.ast_scan import collect_py_files, scan_file
from quant_fund.leakage.rules import RULE_REGISTRY
from quant_fund.proofcore.contracts import (
    LeakageFinding,
    LeakageReport,
    sha256_hex_json,
)


def scan_paths(paths: list[Path], *, rules: set[str] | None = None) -> LeakageReport:
    """Walk .py files, parse with stdlib ast, apply enabled rules, emit report.

    Never raises on parse errors: an unparseable file yields an LH012
    (parse failure, warning) finding instead.
    """
    if rules is not None:
        unknown = rules - set(RULE_REGISTRY)
        if unknown:
            raise ValueError(f"unknown leakage rule id(s): {sorted(unknown)}")
    files = collect_py_files(paths)
    findings: list[LeakageFinding] = []
    for path in files:
        raw = scan_file(path, rules=rules)
        lines: list[str] = []
        if raw:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                lines = []
        for f in raw:
            spec = RULE_REGISTRY[f.rule_id]
            snippet = lines[f.line - 1].strip() if 1 <= f.line <= len(lines) else ""
            findings.append(
                LeakageFinding(
                    rule_id=f.rule_id,
                    severity=spec.severity,
                    path=path.as_posix(),
                    line=f.line,
                    col=f.col,
                    message=f"{spec.title}: {f.message}. {spec.detail}",
                    snippet=snippet[:200],
                )
            )
    errors = sum(1 for f in findings if f.severity == "error")
    warnings = sum(1 for f in findings if f.severity == "warning")
    created_utc = datetime.now(UTC).isoformat()
    payload = {
        "created_utc": created_utc,
        "scanned_files": len(files),
        "findings": [f.model_dump(mode="json") for f in findings],
        "errors": errors,
        "warnings": warnings,
    }
    return LeakageReport(
        created_utc=created_utc,
        scanned_files=len(files),
        findings=findings,
        errors=errors,
        warnings=warnings,
        report_sha256=sha256_hex_json(payload),
    )


def report_to_json(report: LeakageReport) -> str:
    """Canonical pretty JSON for CLI/artifact output."""
    return report.model_dump_json(indent=2)


def report_to_text(report: LeakageReport) -> str:
    """Human-readable scan summary (CLI default)."""
    lines = [
        f"leakage scan: {report.scanned_files} files, "
        f"{report.errors} error(s), {report.warnings} warning(s)",
        f"report_sha256={report.report_sha256}",
    ]
    for f in report.findings:
        lines.append(f"{f.severity.upper():7s} {f.rule_id} {f.path}:{f.line}:{f.col} {f.message}")
        if f.snippet:
            lines.append(f"        | {f.snippet}")
    return "\n".join(lines)
