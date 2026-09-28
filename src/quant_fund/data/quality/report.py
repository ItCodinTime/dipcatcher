"""Receipt-style JSON serialization for quality reports.

Same convention as ``lakehouse.quality.report_json``: pretty-printed,
key-sorted JSON with a ``schema`` tag and a trailing newline. The payload is
deterministic — no wall-clock fields — so ``report_sha256`` is a stable
content address for lineage and receipts.
"""

from __future__ import annotations

import json
from pathlib import Path

from quant_fund.data.quality.models import QualityReport
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes


def report_json(report: QualityReport) -> str:
    """Deterministic JSON rendering (sorted keys, trailing newline)."""
    payload = report.model_dump(mode="json", by_alias=True)
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def report_sha256(report: QualityReport) -> str:
    """Content hash of the canonical report payload, for receipt citation."""
    payload = report.model_dump(mode="python", by_alias=True)
    return hash_bytes(canonical_json_bytes(payload))


def write_report(report: QualityReport, path: Path) -> Path:
    """Write ``report_json`` to ``path`` (parents created); returns the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_json(report), encoding="utf-8")
    return path
