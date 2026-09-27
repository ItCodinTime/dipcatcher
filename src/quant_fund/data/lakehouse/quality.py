"""Fail-closed bar quality checks.

Structural failures (duplicate keys, time moving backwards, OHLC envelope)
default to a zero tolerance. Gap, outlier, and stale checks use the thresholds
in :class:`QualityThresholds`. A Git LFS pointer is not a bar file: the report
fails closed because the payload is absent.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from quant_fund.schemas.errors import DataContractError
from quant_fund.utils.reproducibility import content_address

QUALITY_SCHEMA = "dipcatcher.lake.quality.v1"
_OHLC = ("open", "high", "low", "close")
_EXAMPLE_COLUMNS = (
    "symbol",
    "security_id",
    "ticker",
    "event_time",
    "timestamp",
    "asof",
    "open",
    "high",
    "low",
    "close",
)


class QualityThresholdError(DataContractError):
    """One or more configured quality thresholds were exceeded."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        failed = [
            str(check["name"]) for check in report.get("checks", []) if not check.get("passed")
        ]
        super().__init__("quality thresholds failed: " + ", ".join(failed))


@dataclass(frozen=True)
class QualityThresholds:
    max_duplicate_keys: int = 0
    max_non_monotone: int = 0
    max_ohlc_violations: int = 0
    max_gap: timedelta | None = None
    max_gaps: int = 0
    max_abs_log_return: float = 0.5
    max_outliers: int = 0
    stale_run_length: int = 5
    max_stale_runs: int = 0


def quality_report(
    frame: pl.DataFrame,
    thresholds: QualityThresholds | None = None,
    *,
    enforce: bool = False,
) -> dict[str, Any]:
    """Score one frame. ``enforce=True`` raises :class:`QualityThresholdError`."""
    limits = thresholds or QualityThresholds()
    symbol = _symbol_column(frame)
    clock = _clock_column(frame)
    checks = [
        _duplicates(frame, symbol, clock, limits),
        _non_monotone(frame, symbol, clock, limits),
        _ohlc(frame, limits),
        _gaps(frame, symbol, clock, limits),
        _outliers(frame, symbol, clock, limits),
        _stale(frame, symbol, clock, limits),
    ]
    report = {
        "schema": QUALITY_SCHEMA,
        "rows": frame.height,
        "passed": all(bool(check["passed"]) for check in checks),
        "thresholds": _threshold_dict(limits),
        "checks": checks,
    }
    if enforce and not report["passed"]:
        raise QualityThresholdError(report)
    return report


def quality_report_path(
    path: Path,
    thresholds: QualityThresholds | None = None,
    *,
    enforce: bool = False,
) -> dict[str, Any]:
    """Score a Parquet file, or fail closed on a Git LFS pointer."""
    address = content_address(path)
    if address.kind == "lfs_pointer":
        report = {
            "schema": QUALITY_SCHEMA,
            "path": str(path),
            "content_sha256": address.content_sha256,
            "stored_sha256": address.stored_sha256,
            "kind": "lfs_pointer",
            "declared_size": address.declared_size,
            "rows": None,
            "passed": False,
            "thresholds": _threshold_dict(thresholds or QualityThresholds()),
            "checks": [
                {
                    "name": "payload_present",
                    "applicable": True,
                    "passed": False,
                    "count": 0,
                    "threshold": 1,
                    "detail": "Git LFS pointer; the payload bytes are not in the worktree",
                    "examples": [],
                }
            ],
        }
        if enforce:
            raise QualityThresholdError(report)
        return report
    frame = pl.read_parquet(path)
    report = quality_report(frame, thresholds, enforce=False)
    report["path"] = str(path)
    report["content_sha256"] = address.content_sha256
    report["stored_sha256"] = address.stored_sha256
    report["kind"] = address.kind
    if enforce and not report["passed"]:
        raise QualityThresholdError(report)
    return report


def report_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"


def _duplicates(
    frame: pl.DataFrame,
    symbol: str | None,
    clock: str | None,
    limits: QualityThresholds,
) -> dict[str, Any]:
    if symbol is None or clock is None:
        return _not_applicable("duplicates", "no symbol/time key")
    keys = frame.select(symbol, clock)
    dupes = keys.filter(keys.is_duplicated())
    count = dupes.height
    return _check(
        "duplicates",
        count=count,
        threshold=limits.max_duplicate_keys,
        passed=count <= limits.max_duplicate_keys,
        examples=_examples(frame.filter(keys.is_duplicated())),
        detail="duplicate symbol/time keys",
    )


def _non_monotone(
    frame: pl.DataFrame,
    symbol: str | None,
    clock: str | None,
    limits: QualityThresholds,
) -> dict[str, Any]:
    if symbol is None or clock is None:
        return _not_applicable("non_monotone", "no symbol/time key")
    ordered = frame.with_row_index("_row")
    stamped = ordered.with_columns(pl.col(clock).diff().over(symbol).alias("_delta"))
    bad = stamped.filter(pl.col("_delta") < pl.duration(microseconds=0))
    count = bad.height
    return _check(
        "non_monotone",
        count=count,
        threshold=limits.max_non_monotone,
        passed=count <= limits.max_non_monotone,
        examples=_examples(bad.drop(["_row", "_delta"], strict=False)),
        detail="event time moves backwards within a symbol in stored order",
    )


def _ohlc(frame: pl.DataFrame, limits: QualityThresholds) -> dict[str, Any]:
    if any(name not in frame.columns for name in _OHLC):
        return _not_applicable("ohlc", "OHLC columns are absent")
    bad_expr = pl.lit(False)
    for name in _OHLC:
        finite = pl.col(name).is_finite().fill_null(False)
        bad_expr = bad_expr | pl.col(name).is_null() | ~finite | (pl.col(name).fill_null(0) <= 0)
    bad_expr = (
        bad_expr
        | (pl.col("high") < pl.col("low")).fill_null(False)
        | (pl.col("open") < pl.col("low")).fill_null(False)
        | (pl.col("open") > pl.col("high")).fill_null(False)
        | (pl.col("close") < pl.col("low")).fill_null(False)
        | (pl.col("close") > pl.col("high")).fill_null(False)
    )
    bad = frame.filter(bad_expr)
    count = bad.height
    return _check(
        "ohlc",
        count=count,
        threshold=limits.max_ohlc_violations,
        passed=count <= limits.max_ohlc_violations,
        examples=_examples(bad),
        detail="non-positive prices or open/close outside [low, high]",
    )


def _gaps(
    frame: pl.DataFrame,
    symbol: str | None,
    clock: str | None,
    limits: QualityThresholds,
) -> dict[str, Any]:
    if symbol is None or clock is None:
        return _not_applicable("gaps", "no symbol/time key")
    ordered = frame.sort([symbol, clock]).with_columns(
        pl.col(clock).diff().over(symbol).alias("_delta")
    )
    observed = ordered.select(pl.col("_delta").max()).item()
    observed_seconds = _seconds(observed)
    if limits.max_gap is None:
        return _check(
            "gaps",
            count=None,
            threshold=None,
            passed=True,
            examples=[],
            detail="max_gap is unset; the observed gap is reported and not gated",
            extra={"max_observed_seconds": observed_seconds},
        )
    gap = pl.duration(microseconds=int(limits.max_gap.total_seconds() * 1_000_000))
    bad = ordered.filter(pl.col("_delta") > gap)
    count = bad.height
    return _check(
        "gaps",
        count=count,
        threshold=limits.max_gaps,
        passed=count <= limits.max_gaps,
        examples=_examples(bad.drop("_delta")),
        detail=f"gaps longer than {limits.max_gap}",
        extra={"max_observed_seconds": observed_seconds},
    )


def _outliers(
    frame: pl.DataFrame,
    symbol: str | None,
    clock: str | None,
    limits: QualityThresholds,
) -> dict[str, Any]:
    if symbol is None or clock is None or "close" not in frame.columns:
        return _not_applicable("outliers", "close and a symbol/time key are required")
    ordered = frame.sort([symbol, clock]).with_columns(
        (pl.col("close") / pl.col("close").shift(1).over(symbol)).log().alias("_log_ret")
    )
    finite = ordered.filter(pl.col("_log_ret").is_finite())
    observed = finite.select(pl.col("_log_ret").abs().max()).item() if finite.height else None
    bad = finite.filter(pl.col("_log_ret").abs() > limits.max_abs_log_return)
    count = bad.height
    return _check(
        "outliers",
        count=count,
        threshold=limits.max_outliers,
        passed=count <= limits.max_outliers,
        examples=_examples(bad.drop("_log_ret")),
        detail=f"absolute log return above {limits.max_abs_log_return}",
        extra={"max_abs_log_return_observed": None if observed is None else float(observed)},
    )


def _stale(
    frame: pl.DataFrame,
    symbol: str | None,
    clock: str | None,
    limits: QualityThresholds,
) -> dict[str, Any]:
    if symbol is None or clock is None or "close" not in frame.columns:
        return _not_applicable("stale_prices", "close and a symbol/time key are required")
    if limits.stale_run_length < 2:
        raise DataContractError("stale_run_length must be at least 2")
    ordered = frame.sort([symbol, clock]).with_columns(
        (pl.col("close") != pl.col("close").shift(1).over(symbol)).fill_null(True).alias("_changed")
    )
    runs = (
        ordered.with_columns(pl.col("_changed").cum_sum().over(symbol).alias("_run"))
        .group_by([symbol, "_run"])
        .agg(pl.len().alias("run_length"), pl.col(clock).min().alias(clock))
    )
    bad = runs.filter(pl.col("run_length") >= limits.stale_run_length)
    count = bad.height
    longest = runs.select(pl.col("run_length").max()).item() if runs.height else 0
    return _check(
        "stale_prices",
        count=count,
        threshold=limits.max_stale_runs,
        passed=count <= limits.max_stale_runs,
        examples=_examples(bad),
        detail=f"unchanged close for at least {limits.stale_run_length} bars",
        extra={"max_run_length": int(longest or 0)},
    )


def _symbol_column(frame: pl.DataFrame) -> str | None:
    for name in ("symbol", "security_id", "ticker"):
        if name in frame.columns:
            return name
    return None


def _clock_column(frame: pl.DataFrame) -> str | None:
    for name in ("event_time", "timestamp", "asof"):
        if name in frame.columns:
            return name
    return None


def _check(
    name: str,
    *,
    count: int | None,
    threshold: int | None,
    passed: bool,
    examples: list[dict[str, Any]],
    detail: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": name,
        "applicable": True,
        "passed": passed,
        "count": count,
        "threshold": threshold,
        "detail": detail,
        "examples": examples,
    }
    if extra:
        body.update(extra)
    return body


def _not_applicable(name: str, detail: str) -> dict[str, Any]:
    return {
        "name": name,
        "applicable": False,
        "passed": True,
        "count": None,
        "threshold": None,
        "detail": detail,
        "examples": [],
    }


def _examples(frame: pl.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    if frame.is_empty():
        return []
    columns = [name for name in _EXAMPLE_COLUMNS if name in frame.columns]
    if not columns:
        columns = list(frame.columns)[:6]
    rows: list[dict[str, Any]] = []
    for raw in frame.select(columns).head(limit).to_dicts():
        rows.append({key: _jsonable(value) for key, value in raw.items()})
    return rows


def _jsonable(value: object) -> object:
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, float) and value != value:
        return None
    return value


def _seconds(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, timedelta):
        return value.total_seconds()
    total = getattr(value, "total_seconds", None)
    if callable(total):
        try:
            return float(total())
        except (TypeError, ValueError):
            return None
    return None


def _threshold_dict(limits: QualityThresholds) -> dict[str, Any]:
    payload = asdict(limits)
    gap = payload.get("max_gap")
    if isinstance(gap, timedelta):
        payload["max_gap_seconds"] = gap.total_seconds()
    payload["max_gap"] = None if gap is None else str(gap)
    return payload
