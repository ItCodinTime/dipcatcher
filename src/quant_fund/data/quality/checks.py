"""Individual checks behind :func:`quant_fund.data.quality.score_bars`.

Structural checks (duplicates, non-monotone, OHLC envelope, stale runs,
abs-log-return outliers) are delegated to
:mod:`quant_fund.data.lakehouse.quality` and translated into
:class:`CheckResult`s — this package composes that gate instead of forking
it. The checks implemented here are the ones the lakehouse report does not
cover: non-finite cells, timezone-naive clocks, non-positive volume,
MAD-z price outliers, and missing bars vs expected spacing.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import polars.selectors as cs

from quant_fund.data.quality.models import CheckResult, Offender, QualityConfig

# Structural checks carry more weight than advisory ones in the score.
CHECK_WEIGHTS: dict[str, float] = {
    "duplicates": 3.0,
    "non_monotone": 3.0,
    "ohlc": 3.0,
    "non_finite": 3.0,
    "timezone": 2.0,
    "volume": 2.0,
    "outliers": 2.0,
    "mad_outliers": 2.0,
    "missing_bars": 2.0,
    "stale_prices": 1.0,
}

_LAKEHOUSE_CHECKS = ("duplicates", "non_monotone", "ohlc", "outliers", "stale_prices")
_MAD_FLOOR = 1e-12
_SYMBOL_CANDIDATES = ("symbol", "security_id", "ticker")
_CLOCK_CANDIDATES = ("event_time", "timestamp", "asof")


def resolve_columns(frame: pl.DataFrame, config: QualityConfig) -> tuple[str | None, str | None]:
    """Resolve ``(symbol, clock)`` column names using lakehouse conventions."""
    symbol = config.symbol_column
    if symbol is None:
        symbol = next((name for name in _SYMBOL_CANDIDATES if name in frame.columns), None)
    clock = config.clock_column
    if clock is None:
        clock = next((name for name in _CLOCK_CANDIDATES if name in frame.columns), None)
    return symbol, clock


def translate_lakehouse_checks(
    lakehouse_report: dict[str, Any],
    frame: pl.DataFrame,
    symbol: str | None,
    clock: str | None,
    config: QualityConfig,
) -> list[CheckResult]:
    """Map ``lakehouse.quality.quality_report`` check dicts to CheckResults."""
    results: list[CheckResult] = []
    for check in lakehouse_report.get("checks", []):
        name = str(check.get("name"))
        if name not in _LAKEHOUSE_CHECKS:
            # ``gaps`` needs a configured max_gap; missing_bars owns spacing.
            continue
        applicable = bool(check.get("applicable"))
        count = check.get("count")
        violations = int(count) if count is not None else 0
        detail = str(check.get("detail", ""))
        extras = {
            key: check[key]
            for key in ("max_run_length", "max_abs_log_return_observed")
            if key in check
        }
        results.append(
            CheckResult(
                name=name,
                applicable=applicable,
                violations=violations,
                checked=frame.height if applicable else 0,
                weight=CHECK_WEIGHTS.get(name, 1.0),
                detail=detail,
                worst=_offenders_from_examples(
                    check.get("examples", []), symbol, clock, detail, extras, config.max_offenders
                ),
            )
        )
    return results


def _offenders_from_examples(
    examples: Any,
    symbol: str | None,
    clock: str | None,
    detail: str,
    extra_fields: dict[str, Any],
    limit: int,
) -> list[Offender]:
    if not isinstance(examples, list):
        return []
    offenders: list[Offender] = []
    for row in examples[: max(0, limit)]:
        if not isinstance(row, dict):
            continue
        extra = {key: value for key, value in row.items() if key not in {symbol, clock}}
        extra.update(extra_fields)
        offenders.append(
            Offender(
                timestamp=None if clock is None else _stringify(row.get(clock)),
                symbol=None if symbol is None else _stringify(row.get(symbol)),
                detail=detail,
                extra=extra,
            )
        )
    return offenders


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def check_non_finite(frame: pl.DataFrame, config: QualityConfig) -> CheckResult:
    """Rows with a null or non-finite (NaN/±inf) value in a numeric column."""
    name = "non_finite"
    numeric = frame.select(cs.numeric())
    if not numeric.columns or frame.height == 0:
        return _na(name, "no numeric columns")
    bad_expr = pl.lit(False)
    for column in numeric.columns:
        bad_expr = bad_expr | pl.col(column).is_null() | ~pl.col(column).is_finite()
    bad = frame.filter(bad_expr.fill_null(False))
    per_column = {
        column: int(
            frame.filter(
                (pl.col(column).is_null() | ~pl.col(column).is_finite()).fill_null(False)
            ).height
        )
        for column in numeric.columns
    }
    offenders = _row_offenders(bad, frame, config, name, "non-finite or null numeric cell")
    return CheckResult(
        name=name,
        applicable=True,
        violations=bad.height,
        checked=frame.height,
        weight=CHECK_WEIGHTS[name],
        detail=f"rows with a null/NaN/inf value in a numeric column (per-column: {per_column})",
        worst=offenders,
    )


def check_timezone(frame: pl.DataFrame, clock: str | None, config: QualityConfig) -> CheckResult:
    """Flag a clock column whose timestamps cannot or do not carry a timezone."""
    name = "timezone"
    if clock is None or clock not in frame.columns:
        return _na(name, "no clock column")
    dtype = frame.schema[clock]
    if isinstance(dtype, pl.Datetime):
        if dtype.time_zone is not None:
            return CheckResult(
                name=name,
                applicable=True,
                violations=0,
                checked=frame.height,
                weight=CHECK_WEIGHTS[name],
                detail=f"clock dtype {dtype} is timezone-aware",
            )
        offenders = _row_offenders(frame, frame, config, name, "timezone-naive timestamp")
        return CheckResult(
            name=name,
            applicable=True,
            violations=frame.height,
            checked=frame.height,
            weight=CHECK_WEIGHTS[name],
            detail=f"clock dtype {dtype} has no timezone",
            worst=offenders,
        )
    if dtype == pl.Date:
        return CheckResult(
            name=name,
            applicable=True,
            violations=frame.height,
            checked=frame.height,
            weight=CHECK_WEIGHTS[name],
            detail="Date-typed clock cannot carry a timezone",
            worst=_row_offenders(frame, frame, config, name, "date-typed clock"),
        )
    return _na(name, f"clock dtype {dtype} is not temporal")


def check_volume(frame: pl.DataFrame, config: QualityConfig) -> CheckResult:
    """Rows with zero or negative volume."""
    name = "volume"
    if "volume" not in frame.columns:
        return _na(name, "no volume column")
    if not frame.schema["volume"].is_numeric():
        return _na(name, f"volume dtype {frame.schema['volume']} is not numeric")
    present = frame.filter(pl.col("volume").is_not_null())
    bad = present.filter(pl.col("volume") <= 0)
    return CheckResult(
        name=name,
        applicable=True,
        violations=bad.height,
        checked=present.height,
        weight=CHECK_WEIGHTS[name],
        detail="rows with volume <= 0 (null volume is reported by non_finite)",
        worst=_row_offenders(bad, frame, config, name, "volume <= 0"),
    )


def check_mad_outliers(
    frame: pl.DataFrame,
    symbol: str | None,
    clock: str | None,
    config: QualityConfig,
) -> CheckResult:
    """Close-to-close log returns flagged by the robust modified z-score.

    z = 0.6745 * (r - median(r)) / max(MAD(r), 1e-12) over the whole
    dataset; |z| >= ``config.mad_z_threshold`` is a violation (Iglewicz &
    Hoaglin cutoff 3.5). The 1e-12 MAD floor keeps floating-point noise on
    near-constant returns from reading as outliers while still flagging any
    economically meaningful deviation when MAD is exactly zero.
    """
    name = "mad_outliers"
    if clock is None or "close" not in frame.columns:
        return _na(name, "close and a clock column are required")
    if not frame.schema["close"].is_numeric():
        return _na(name, f"close dtype {frame.schema['close']} is not numeric")
    keys = [symbol, clock] if symbol else [clock]
    ordered = frame.sort(keys)
    shift = pl.col("close").shift(1)
    if symbol:
        shift = shift.over(symbol)
    ordered = ordered.with_columns((pl.col("close") / shift).log().alias("_log_ret"))
    finite = ordered.filter(pl.col("_log_ret").is_finite())
    if finite.height == 0:
        return CheckResult(
            name=name,
            applicable=True,
            violations=0,
            checked=0,
            weight=CHECK_WEIGHTS[name],
            detail="no finite returns to score",
        )
    median = _as_float(finite["_log_ret"].median())
    deviations = (finite["_log_ret"] - median).abs()
    mad = _as_float(deviations.median())
    scale = max(mad, _MAD_FLOOR)
    scored = finite.with_columns((0.6745 * (pl.col("_log_ret") - median) / scale).alias("_z"))
    bad = scored.filter(pl.col("_z").abs() >= config.mad_z_threshold)
    ranked = bad.sort(pl.col("_z").abs(), descending=True)
    detail = (
        f"|modified z| >= {config.mad_z_threshold} on log returns "
        f"(median={median:.6g}, MAD={mad:.6g}, scale={scale:.6g})"
    )
    offenders = [
        Offender(
            timestamp=_stringify(row.get(clock)),
            symbol=None if symbol is None else _stringify(row.get(symbol)),
            detail="log-return MAD outlier",
            value=float(row["_z"]),
            extra={"log_return": row["_log_ret"], "z": row["_z"]},
        )
        for row in ranked.head(config.max_offenders).to_dicts()
    ]
    return CheckResult(
        name=name,
        applicable=True,
        violations=bad.height,
        checked=finite.height,
        weight=CHECK_WEIGHTS[name],
        detail=detail,
        worst=offenders,
    )


def check_missing_bars(
    frame: pl.DataFrame,
    symbol: str | None,
    clock: str | None,
    config: QualityConfig,
) -> CheckResult:
    """Adjacent-bar spacing wider than the expected interval.

    Expected interval is ``config.expected_interval_seconds`` when set, else
    the per-symbol (or dataset-wide when no symbol column) median of positive
    spacings. A spacing larger than ``expected * gap_factor`` counts as a
    gap event; the estimated missing-bar count is
    ``round(spacing / expected) - 1``. This is an interval heuristic only —
    it does not know sessions or holidays; the calendars track owns trading
    calendars.
    """
    name = "missing_bars"
    if clock is None or clock not in frame.columns:
        return _na(name, "no clock column")
    keys = [symbol, clock] if symbol else [clock]
    ordered = frame.sort(keys).with_row_index("_row")
    diff = pl.col(clock).diff()
    if symbol:
        diff = diff.over(symbol)
    ordered = ordered.with_columns(diff.alias("_diff"))
    diff_dtype = ordered.schema["_diff"]
    if isinstance(diff_dtype, pl.Duration):
        spacing_expr = pl.col("_diff").dt.total_microseconds() / 1_000_000.0
    elif diff_dtype.is_numeric():
        spacing_expr = pl.col("_diff").cast(pl.Float64, strict=False)
    else:
        return _na(name, f"clock dtype {frame.schema[clock]} diffs are not measurable")
    ordered = ordered.with_columns(spacing_expr.alias("_spacing"))
    positive = ordered.filter(pl.col("_spacing") > 0)
    if positive.height == 0:
        return CheckResult(
            name=name,
            applicable=True,
            violations=0,
            checked=0,
            weight=CHECK_WEIGHTS[name],
            detail="fewer than two distinct timestamps",
        )
    if config.expected_interval_seconds is not None:
        expected_by_group: dict[Any, float] = {}
        default_expected = config.expected_interval_seconds
    elif symbol:
        medians = positive.group_by(symbol).agg(pl.col("_spacing").median().alias("_expected"))
        expected_by_group = {row[symbol]: _as_float(row["_expected"]) for row in medians.to_dicts()}
        default_expected = _as_float(positive["_spacing"].median())
    else:
        expected_by_group = {}
        default_expected = _as_float(positive["_spacing"].median())
    if default_expected <= 0 and not expected_by_group:
        return CheckResult(
            name=name,
            applicable=True,
            violations=0,
            checked=positive.height,
            weight=CHECK_WEIGHTS[name],
            detail="could not infer a positive expected interval",
        )

    flag_rows: list[dict[str, Any]] = []
    total_missing = 0
    for row in positive.to_dicts():
        group_key = row[symbol] if symbol is not None else None
        expected = (
            config.expected_interval_seconds
            if config.expected_interval_seconds is not None
            else expected_by_group.get(group_key, default_expected)
        )
        if expected is None or expected <= 0:
            continue
        spacing = float(row["_spacing"])
        if spacing > expected * config.gap_factor:
            missing = max(1, int(round(spacing / expected)) - 1)
            total_missing += missing
            flag_rows.append({**row, "_expected": expected, "_missing": missing})
    flag_rows.sort(key=lambda row: row["_spacing"], reverse=True)
    offenders = [
        Offender(
            timestamp=_stringify(row.get(clock)),
            symbol=None if symbol is None else _stringify(row.get(symbol)),
            detail=(
                f"gap of {row['_spacing']:.6g} vs expected {row['_expected']:.6g}; "
                f"~{row['_missing']} missing bar(s)"
            ),
            value=float(row["_spacing"]),
            extra={"expected_interval": row["_expected"], "estimated_missing": row["_missing"]},
        )
        for row in flag_rows[: config.max_offenders]
    ]
    expected_desc = (
        f"{config.expected_interval_seconds:.6g} (configured)"
        if config.expected_interval_seconds is not None
        else f"{default_expected:.6g} (median spacing)"
    )
    return CheckResult(
        name=name,
        applicable=True,
        violations=len(flag_rows),
        checked=positive.height,
        weight=CHECK_WEIGHTS[name],
        detail=(
            f"spacings beyond {config.gap_factor}x expected interval {expected_desc}; "
            f"~{total_missing} missing bar(s) total"
        ),
        worst=offenders,
    )


def _as_float(value: Any) -> float:
    """Narrow a polars scalar (median/diff) to float; non-numerics become 0.0."""
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _row_offenders(
    bad: pl.DataFrame,
    frame: pl.DataFrame,
    config: QualityConfig,
    name: str,
    detail: str,
) -> list[Offender]:
    symbol, clock = config.symbol_column, config.clock_column
    if symbol is None:
        symbol = next((c for c in _SYMBOL_CANDIDATES if c in frame.columns), None)
    if clock is None:
        clock = next((c for c in _CLOCK_CANDIDATES if c in frame.columns), None)
    offenders: list[Offender] = []
    for row in bad.head(config.max_offenders).to_dicts():
        offenders.append(
            Offender(
                timestamp=None if clock is None else _stringify(row.get(clock)),
                symbol=None if symbol is None else _stringify(row.get(symbol)),
                detail=detail,
            )
        )
    return offenders


def _na(name: str, detail: str) -> CheckResult:
    return CheckResult(
        name=name,
        applicable=False,
        violations=0,
        checked=0,
        weight=CHECK_WEIGHTS.get(name, 1.0),
        detail=detail,
    )
