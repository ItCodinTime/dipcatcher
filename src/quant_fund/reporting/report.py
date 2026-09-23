"""Markdown research reports. Synthetic runs are labeled."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def build_evidence_report(
    *,
    candidates: dict[str, Any],
    provenance: dict[str, Any] | None = None,
    health: dict[str, Any] | None = None,
    promotion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a conservative, source-backed model evidence report.

    Missing sections are recorded as warnings; the function never infers
    readiness from absent metrics and never changes promotion state.
    """
    warnings: list[str] = []
    if not candidates:
        warnings.append("candidate_metrics_missing")
    if not provenance:
        warnings.append("provenance_missing")
    if not health:
        warnings.append("health_report_missing")
    if not promotion:
        warnings.append("promotion_receipt_missing")
    artifact_hash = (provenance or {}).get("artifact_sha256")
    if not isinstance(artifact_hash, str) or len(artifact_hash) != 64:
        warnings.append("artifact_hash_missing_or_invalid")
    if (provenance or {}).get("manifest_valid") is not True:
        warnings.append("artifact_manifest_not_verified")
    if isinstance(promotion, dict) and promotion.get("promote") is not True:
        warnings.append("promotion_not_approved")
    if isinstance(health, dict) and health.get("status") != "ok":
        warnings.append("health_not_ok")
    synthetic = str((provenance or {}).get("data_source", "")).upper() == "SYNTHETIC"
    if synthetic:
        warnings.append("synthetic_evidence_not_promotable")
    return {
        "schema": "evidence_report.v1",
        "candidates": candidates,
        "provenance": provenance or {},
        "health": health or {},
        "promotion": promotion or {},
        "warnings": sorted(set(warnings)),
        "status": "insufficient_evidence" if warnings else "complete",
        "research_only": True,
        "live_pnl_claim": False,
    }


def write_evidence_report(
    root: Path,
    *,
    candidates: dict[str, Any],
    provenance: dict[str, Any] | None = None,
    health: dict[str, Any] | None = None,
    promotion: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Persist JSON and Markdown evidence reports atomically."""
    report = build_evidence_report(
        candidates=candidates,
        provenance=provenance,
        health=health,
        promotion=promotion,
    )
    directory = latest_report_dir(Path(root))
    paths = {
        "json": directory / "evidence_report.json",
        "markdown": directory / "evidence_report.md",
    }
    markdown = write_report_text(report)
    for key, content in (("json", json.dumps(report, indent=2, sort_keys=True) + "\n"), ("markdown", markdown)):
        destination = paths[key]
        with NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", mode="w", encoding="utf-8", delete=False
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, destination)
    digest = hashlib.sha256(paths["json"].read_bytes()).hexdigest()
    sidecar = paths["json"].with_name(f"{paths['json'].name}.sha256")
    with NamedTemporaryFile(
        dir=sidecar.parent, prefix=f".{sidecar.name}.", suffix=".tmp", mode="w", encoding="ascii", delete=False
    ) as temporary:
        temporary.write(digest + "\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, sidecar)
    return paths


def write_report_text(report: dict[str, Any]) -> str:
    """Render an evidence report without implying readiness."""
    lines = ["# Institutional Evidence Report", "", f"- status: {report['status']}", "- research_only: true", "- live_pnl_claim: false", ""]
    lines.append("## Warnings")
    warnings = report.get("warnings") or ["none"]
    lines.extend(f"- {warning}" for warning in warnings)
    lines.append("")
    for section in ("candidates", "provenance", "health", "promotion"):
        lines.append(f"## {section.title()}")
        lines.append("```json")
        lines.append(json.dumps(report.get(section, {}), indent=2, sort_keys=True, default=str))
        lines.extend(["```", ""])
    return "\n".join(lines)


def write_report(path: Path, title: str, sections: dict[str, Any], *, synthetic: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if synthetic:
        lines += ["> SYNTHETIC DATA. Not evidence of live profitability.", ""]
    for name, body in sections.items():
        lines.append(f"## {name}")
        if isinstance(body, dict):
            for k, v in body.items():
                if isinstance(v, float):
                    lines.append(f"- {k}: {v:.6g}")
                else:
                    lines.append(f"- {k}: {v}")
        else:
            lines.append(str(body))
        lines.append("")
    path.write_text("\n".join(lines))
    return path


def latest_report_dir(root: Path) -> Path:
    d = root / "metadata" / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d
