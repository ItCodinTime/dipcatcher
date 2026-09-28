"""Typed public API for the dipcatcher research harness.

Import from the package root (preferred) or from this module::

    from quant_fund import (
        load_config,
        ingest,
        run_backtest,
        run_research,
        read_research_receipt,
        verify_research,
    )

``load_config`` reads a YAML experiment into :class:`AppConfig`. ``ingest``
writes the point-in-time lake and returns artifact paths. ``run_backtest``
replays target weights against daily bars. ``run_research`` runs the research
harness and returns a typed snapshot of the notebook. ``read_research_receipt``
loads a persisted receipt. ``verify_research`` fail-closed-checks one.

These entry points do not place orders and do not claim live profit. The HTTP
service lives in :mod:`quant_fund.api` and is not part of this surface.
Execution engines stay importable from their packages; this module is the
stable contract.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

import polars as pl

from quant_fund.backtest.engine import (
    StaleValuationError,
)
from quant_fund.backtest.engine import (
    run_backtest as _run_backtest,
)
from quant_fund.config.loader import load_config as _load_config
from quant_fund.config.models import AppConfig
from quant_fund.data.ingest import ingest as _ingest
from quant_fund.research.agent import HypothesisResult, ResearchNotebook
from quant_fund.research.agent import run_research as _run_research
from quant_fund.research.verify import verify_research_artifact
from quant_fund.risk.overlay import BookRiskOverlay

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]

__all__ = [
    "AppConfig",
    "BacktestMetrics",
    "BacktestRun",
    "BookRiskOverlay",
    "HypothesisResult",
    "IngestPaths",
    "JsonValue",
    "ResearchReceipt",
    "ResearchRun",
    "ResearchVerification",
    "StaleValuationError",
    "ingest",
    "load_config",
    "read_research_receipt",
    "run_backtest",
    "run_research",
    "verify_research",
]


class IngestPaths(TypedDict):
    """Parquet and manifest paths written by :func:`ingest`."""

    bars: Path
    actions: Path
    master: Path
    silver: Path
    universe: Path
    manifest: Path


class BacktestMetrics(TypedDict):
    """Execution and control diagnostics from a simulated replay.

    These are not proper research scores or evidence of trading performance.
    The engine's accounting frames remain available on :class:`BacktestRun`;
    performance summaries are deliberately absent from this public mapping.
    """

    n: int
    risk_gate_rejects: int
    cash_rejects: int
    kill_switch_halts: int
    implementation_shortfall: dict[str, object]
    research_only: bool
    live_pnl_claim: bool
    data_source: str
    claim: Literal["execution_diagnostic_only"]
    turnover_bps_cost: float
    garch_risk_overlay_dates: int
    realized_garch_risk_overlay_dates: int
    mean_turnover: NotRequired[float]
    commission: NotRequired[float]
    spread: NotRequired[float]
    impact: NotRequired[float]
    label: NotRequired[str]


class ResearchReceipt(TypedDict):
    """JSON object persisted as a research receipt."""

    schema_version: int
    firm: str
    product: str
    version: str
    generated_at: str
    data_source: str
    synthetic: bool
    disclaimer: str
    ranking_target: str
    claim: str
    families: dict[str, JsonValue]
    rankers: list[dict[str, JsonValue]]
    hypotheses: list[dict[str, JsonValue]]
    scorecard: dict[str, dict[str, JsonValue]]
    provenance: dict[str, JsonValue]
    artifacts: dict[str, str]


class ResearchVerification(TypedDict):
    """Fail-closed check of one research receipt.

    ``run_id``, ``scorecard_families``, and ``claim`` are present once the
    file parses as a notebook object. A receipt that cannot be read carries
    only ``valid``, ``path``, and ``errors``.
    """

    valid: bool
    path: str
    errors: list[str]
    run_id: NotRequired[object]
    scorecard_families: NotRequired[int]
    claim: NotRequired[str]


@dataclass(frozen=True, eq=False)
class BacktestRun:
    """Research backtest output.

    ``equity`` and ``fills`` are the engine's accounting frames. ``metrics``
    contains execution and control diagnostics, not research scores.
    Equality is disabled because frame ``==`` is not a boolean.
    """

    equity: pl.DataFrame
    fills: pl.DataFrame
    metrics: BacktestMetrics
    frictionless: bool
    source_note: str


@dataclass(frozen=True)
class ResearchRun:
    """Typed snapshot of one research-harness notebook.

    Scalar fields and hypothesis rows are the in-memory notebook.
    ``families``, ``rankers``, ``scorecard``, and ``provenance`` match the
    receipt JSON: non-finite floats are null. This is research evidence,
    not a live-trading result.
    """

    schema_version: int
    firm: str
    product: str
    version: str
    generated_at: str
    data_source: str
    synthetic: bool
    disclaimer: str
    ranking_target: str
    claim: str
    families: dict[str, JsonValue]
    rankers: tuple[dict[str, JsonValue], ...]
    hypotheses: tuple[HypothesisResult, ...]
    scorecard: dict[str, dict[str, JsonValue]]
    provenance: dict[str, JsonValue]
    artifacts: dict[str, str]

    def to_receipt(self) -> ResearchReceipt:
        """Return the JSON receipt for this run."""
        return {
            "schema_version": self.schema_version,
            "firm": self.firm,
            "product": self.product,
            "version": self.version,
            "generated_at": self.generated_at,
            "data_source": self.data_source,
            "synthetic": self.synthetic,
            "disclaimer": self.disclaimer,
            "ranking_target": self.ranking_target,
            "claim": self.claim,
            "families": self.families,
            "rankers": list(self.rankers),
            "hypotheses": [_hypothesis_dict(row) for row in self.hypotheses],
            "scorecard": self.scorecard,
            "provenance": self.provenance,
            "artifacts": dict(self.artifacts),
        }


def load_config(path: str | Path) -> AppConfig:
    """Load a YAML experiment, following ``inherit:`` inside the config root."""
    return _load_config(path)


def ingest(config: AppConfig) -> IngestPaths:
    """Ingest the configured market source into the bronze/silver lake.

    Returns the artifact paths. Synthetic sources are labeled by the
    pipeline; this function does not fetch a broker.
    """
    paths = _ingest(config)
    return {
        "bars": paths["bars"],
        "actions": paths["actions"],
        "master": paths["master"],
        "silver": paths["silver"],
        "universe": paths["universe"],
        "manifest": paths["manifest"],
    }


def run_backtest(
    bars: pl.DataFrame,
    weights: pl.DataFrame,
    config: AppConfig,
    *,
    initial_nav: float = 1_000_000.0,
    risk_overlay: BookRiskOverlay | None = None,
    fast: bool | None = None,
) -> BacktestRun:
    """Replay target weights against daily bars.

    ``weights`` columns are ``event_time``, ``security_id``, and
    ``target_weight``. A target computed from the close at ``t`` fills at
    the next open unless the config selects the close auction. The result
    is a research simulation.

    ``fast=True`` pins the vectorized replay and fails closed
    (``ValueError``) on workloads outside its bit-identical class;
    ``fast=False`` pins the reference event loop; ``None`` auto-selects.
    """
    result = _run_backtest(
        bars,
        weights,
        config,
        initial_nav=initial_nav,
        risk_overlay=risk_overlay,
        fast=fast,
    )
    return BacktestRun(
        equity=result.equity,
        fills=result.fills,
        metrics=_as_backtest_metrics(result.metrics),
        frictionless=result.frictionless,
        source_note=result.source_note,
    )


def run_research(config: AppConfig) -> ResearchRun:
    """Run the research harness and return a typed snapshot.

    Receipts on disk are the harness receipts. The returned snapshot does
    not add a live-trading claim.
    """
    return _research_run(_run_research(config))


def read_research_receipt(path: str | Path) -> ResearchReceipt:
    """Load a research receipt JSON object.

    This parses structure. It does not check hashes or honesty rules; use
    :func:`verify_research` for that. Non-finite numbers are not valid JSON
    and raise :class:`json.JSONDecodeError` or :class:`ValueError`.
    """
    receipt_path = Path(path)
    parsed = _as_json(json.loads(receipt_path.read_text(encoding="utf-8")))
    if not isinstance(parsed, dict):
        raise ValueError(f"research receipt {receipt_path} must be a JSON object")
    return _research_receipt(parsed)


def verify_research(path: str | Path) -> ResearchVerification:
    """Validate a persisted research receipt.

    Returns the harness report: ``valid`` is false when any check fails.
    The report does not interpret scores as tradable profit.
    """
    report = verify_research_artifact(Path(path))
    valid = report["valid"]
    report_path = report["path"]
    errors = report["errors"]
    if (
        not isinstance(valid, bool)
        or not isinstance(report_path, str)
        or not isinstance(errors, list)
    ):
        raise TypeError("research verification report has an unexpected shape")
    error_lines: list[str] = []
    for item in errors:
        if not isinstance(item, str):
            raise TypeError("research verification errors must be strings")
        error_lines.append(item)
    typed: ResearchVerification = {
        "valid": valid,
        "path": report_path,
        "errors": error_lines,
    }
    if "run_id" in report:
        typed["run_id"] = report["run_id"]
    if "scorecard_families" in report:
        families = report["scorecard_families"]
        if isinstance(families, bool) or not isinstance(families, int):
            raise TypeError("scorecard_families must be an int")
        typed["scorecard_families"] = families
    if "claim" in report:
        claim = report["claim"]
        if not isinstance(claim, str):
            raise TypeError("verification claim must be a string")
        typed["claim"] = claim
    return typed


def _as_json(value: object) -> JsonValue:
    """Normalize one JSON-like value.

    Non-finite floats become null, matching research receipt serialization.
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        number = float(value)
        if math.isfinite(number):
            return number
        return None
    if isinstance(value, list | tuple):
        return [_as_json(item) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            normalized[key] = _as_json(item)
        return normalized
    raise ValueError(f"expected a JSON value, got {type(value).__name__}")


def _as_object(value: object) -> dict[str, JsonValue]:
    parsed = _as_json(value)
    if not isinstance(parsed, dict):
        raise ValueError("expected a JSON object")
    return parsed


def _require(obj: dict[str, JsonValue], key: str) -> JsonValue:
    if key not in obj:
        raise ValueError(f"research receipt missing {key}")
    return obj[key]


def _require_str(obj: dict[str, JsonValue], key: str) -> str:
    value = _require(obj, key)
    if not isinstance(value, str):
        raise ValueError(f"research receipt field {key} must be a string")
    return value


def _require_int(obj: dict[str, JsonValue], key: str) -> int:
    value = _require(obj, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"research receipt field {key} must be an int")
    return value


def _require_bool(obj: dict[str, JsonValue], key: str) -> bool:
    value = _require(obj, key)
    if not isinstance(value, bool):
        raise ValueError(f"research receipt field {key} must be a bool")
    return value


def _require_object(obj: dict[str, JsonValue], key: str) -> dict[str, JsonValue]:
    value = _require(obj, key)
    if not isinstance(value, dict):
        raise ValueError(f"research receipt field {key} must be an object")
    return value


def _require_dict_list(obj: dict[str, JsonValue], key: str) -> list[dict[str, JsonValue]]:
    value = _require(obj, key)
    if not isinstance(value, list):
        raise ValueError(f"research receipt field {key} must be a list")
    rows: list[dict[str, JsonValue]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"research receipt field {key}[{index}] must be an object")
        rows.append(item)
    return rows


def _require_scorecard(obj: dict[str, JsonValue], key: str) -> dict[str, dict[str, JsonValue]]:
    value = _require_object(obj, key)
    scorecard: dict[str, dict[str, JsonValue]] = {}
    for name, item in value.items():
        if not isinstance(item, dict):
            raise ValueError(f"research receipt scorecard {name} must be an object")
        scorecard[name] = item
    return scorecard


def _require_artifacts(obj: dict[str, JsonValue], key: str) -> dict[str, str]:
    value = _require_object(obj, key)
    artifacts: dict[str, str] = {}
    for name, item in value.items():
        if not isinstance(item, str):
            raise ValueError(f"research receipt artifact {name} must be a string")
        artifacts[name] = item
    return artifacts


def _research_receipt(payload: dict[str, JsonValue]) -> ResearchReceipt:
    return {
        "schema_version": _require_int(payload, "schema_version"),
        "firm": _require_str(payload, "firm"),
        "product": _require_str(payload, "product"),
        "version": _require_str(payload, "version"),
        "generated_at": _require_str(payload, "generated_at"),
        "data_source": _require_str(payload, "data_source"),
        "synthetic": _require_bool(payload, "synthetic"),
        "disclaimer": _require_str(payload, "disclaimer"),
        "ranking_target": _require_str(payload, "ranking_target"),
        "claim": _require_str(payload, "claim"),
        "families": _require_object(payload, "families"),
        "rankers": _require_dict_list(payload, "rankers"),
        "hypotheses": _require_dict_list(payload, "hypotheses"),
        "scorecard": _require_scorecard(payload, "scorecard"),
        "provenance": _require_object(payload, "provenance"),
        "artifacts": _require_artifacts(payload, "artifacts"),
    }


def _hypothesis_dict(row: HypothesisResult) -> dict[str, JsonValue]:
    parsed = _as_json(asdict(row))
    if not isinstance(parsed, dict):
        raise TypeError("hypothesis row must serialize to an object")
    return parsed


def _research_run(notebook: ResearchNotebook) -> ResearchRun:
    receipt = _research_receipt(_as_object(notebook.to_dict()))
    return ResearchRun(
        schema_version=notebook.schema_version,
        firm=notebook.firm,
        product=notebook.product,
        version=notebook.version,
        generated_at=notebook.generated_at,
        data_source=notebook.data_source,
        synthetic=notebook.synthetic,
        disclaimer=notebook.disclaimer,
        ranking_target=notebook.ranking_target,
        claim=notebook.claim,
        families=receipt["families"],
        rankers=tuple(receipt["rankers"]),
        hypotheses=tuple(notebook.hypotheses),
        scorecard=receipt["scorecard"],
        provenance=receipt["provenance"],
        artifacts=dict(notebook.artifacts),
    )


def _as_backtest_metrics(raw: object) -> BacktestMetrics:
    """Expose only execution and control diagnostics from engine metrics."""
    if not isinstance(raw, dict):
        raise TypeError("backtest metrics must be a dict")
    if not all(isinstance(key, str) for key in raw):
        raise TypeError("backtest metric keys must be strings")
    values = cast(dict[str, object], raw)
    if values.get("research_only") is not True:
        raise ValueError("backtest metrics must set research_only=True")
    if values.get("live_pnl_claim") is not False:
        raise ValueError("backtest metrics must set live_pnl_claim=False")
    counts: dict[str, int] = {}
    for key in (
        "n",
        "risk_gate_rejects",
        "cash_rejects",
        "kill_switch_halts",
        "garch_risk_overlay_dates",
        "realized_garch_risk_overlay_dates",
    ):
        value = values.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"backtest metrics {key} must be an int")
        counts[key] = value
    shortfall = values.get("implementation_shortfall")
    if not isinstance(shortfall, dict) or not all(isinstance(key, str) for key in shortfall):
        raise TypeError("implementation_shortfall must be a dict")
    data_source = values.get("data_source")
    if not isinstance(data_source, str):
        raise TypeError("data_source must be a str")
    turnover_bps_cost = values.get("turnover_bps_cost")
    if isinstance(turnover_bps_cost, bool) or not isinstance(turnover_bps_cost, (int, float)):
        raise TypeError("turnover_bps_cost must be a number")
    diagnostics: BacktestMetrics = {
        "n": counts["n"],
        "risk_gate_rejects": counts["risk_gate_rejects"],
        "cash_rejects": counts["cash_rejects"],
        "kill_switch_halts": counts["kill_switch_halts"],
        "implementation_shortfall": cast(dict[str, object], dict(shortfall)),
        "research_only": True,
        "live_pnl_claim": False,
        "data_source": data_source,
        "claim": "execution_diagnostic_only",
        "turnover_bps_cost": float(turnover_bps_cost),
        "garch_risk_overlay_dates": counts["garch_risk_overlay_dates"],
        "realized_garch_risk_overlay_dates": counts["realized_garch_risk_overlay_dates"],
    }
    for key in ("mean_turnover", "commission", "spread", "impact"):
        if key in values:
            number = values[key]
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise TypeError(f"backtest metrics {key} must be a number")
            if key == "mean_turnover":
                diagnostics["mean_turnover"] = float(number)
            elif key == "commission":
                diagnostics["commission"] = float(number)
            elif key == "spread":
                diagnostics["spread"] = float(number)
            else:
                diagnostics["impact"] = float(number)
    if "label" in values:
        label = values["label"]
        if not isinstance(label, str):
            raise TypeError("backtest metrics label must be a str")
        diagnostics["label"] = label
    return diagnostics
