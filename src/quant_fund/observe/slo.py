"""SLO definitions for the observability stack.

The evaluator compares a metrics snapshot to the thresholds in
``deploy/observability/slo.yml``. Missing inputs are ``no_data``, not a pass.
Nothing here claims that a live or research process meets the objectives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from quant_fund.observe.metrics import REGISTRY, histogram_quantile_upper


@dataclass(frozen=True)
class Slo:
    id: str
    description: str
    metric: str
    comparator: str
    threshold: float


@dataclass(frozen=True)
class SloResult:
    id: str
    status: str
    metric: str
    value: float | None
    threshold: float
    comparator: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "comparator": self.comparator,
        }


def default_slo_path() -> Path:
    return Path(__file__).resolve().parents[3] / "deploy" / "observability" / "slo.yml"


def load_slos(path: Path | None = None) -> list[Slo]:
    document = yaml.safe_load((path or default_slo_path()).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("version") != 1:
        raise ValueError("SLO file must be a mapping with version 1")
    rows = document.get("slos")
    if not isinstance(rows, list) or not rows:
        raise ValueError("SLO file has no objectives")
    loaded: list[Slo] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("SLO row must be a mapping")
        comparator = str(row["comparator"])
        if comparator not in {"lt", "le"}:
            raise ValueError(f"unsupported SLO comparator {comparator}")
        loaded.append(
            Slo(
                id=str(row["id"]),
                description=str(row["description"]),
                metric=str(row["metric"]),
                comparator=comparator,
                threshold=float(row["threshold"]),
            )
        )
    return loaded


def current_slo_snapshot() -> dict[str, float]:
    """Summarize the in-process registry for the committed SLO metrics."""
    snap = REGISTRY.snapshot()
    counts = 0
    errors = 0
    p99_values: list[float] = []
    for stage, histogram in snap["histograms"].items():
        counts += int(histogram["n"])
        upper = histogram_quantile_upper(stage, 0.99)
        if math.isfinite(upper):
            p99_values.append(upper)
    for stage, count in snap["errors"].items():
        if stage == "otel_export":
            continue
        errors += int(count)
    snapshot: dict[str, float] = {}
    if counts:
        snapshot["latency_p99_seconds"] = max(p99_values) if p99_values else math.inf
        snapshot["error_ratio"] = errors / counts
    freshness = snap["gauges"].get("dipcatcher_data_freshness_seconds")
    if freshness is not None:
        snapshot["freshness_seconds"] = float(freshness)
    return snapshot


def evaluate(snapshot: dict[str, float], slos: list[Slo]) -> list[SloResult]:
    results: list[SloResult] = []
    for slo in slos:
        if slo.metric not in snapshot or not math.isfinite(snapshot[slo.metric]):
            results.append(
                SloResult(
                    id=slo.id,
                    status="no_data",
                    metric=slo.metric,
                    value=None,
                    threshold=slo.threshold,
                    comparator=slo.comparator,
                )
            )
            continue
        value = snapshot[slo.metric]
        ok = value < slo.threshold if slo.comparator == "lt" else value <= slo.threshold
        results.append(
            SloResult(
                id=slo.id,
                status="pass" if ok else "fail",
                metric=slo.metric,
                value=value,
                threshold=slo.threshold,
                comparator=slo.comparator,
            )
        )
    return results


def objectives_met(results: list[SloResult]) -> bool:
    return bool(results) and all(item.status == "pass" for item in results)
