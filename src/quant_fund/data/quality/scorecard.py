"""``score_bars`` — the per-dataset quality scorecard entry point."""

from __future__ import annotations

from typing import Any

import polars as pl

from quant_fund.data.lakehouse.quality import QualityThresholds
from quant_fund.data.lakehouse.quality import quality_report as lakehouse_quality_report
from quant_fund.data.quality.checks import (
    check_mad_outliers,
    check_missing_bars,
    check_non_finite,
    check_timezone,
    check_volume,
    resolve_columns,
    translate_lakehouse_checks,
)
from quant_fund.data.quality.models import QualityConfig, QualityReport


def score_bars(
    df: Any,
    config: QualityConfig | None = None,
    *,
    dataset: str | None = None,
) -> QualityReport:
    """Score one bar/quote frame and return a structured QualityReport.

    ``df`` may be a ``polars.DataFrame``, ``polars.LazyFrame``, or
    ``pandas.DataFrame``; pandas input is converted via ``pl.from_pandas``.
    The report is deterministic: identical frames and configs produce
    identical JSON.
    """
    cfg = config or QualityConfig()
    frame = _as_polars(df)
    symbol, clock = resolve_columns(frame, cfg)

    thresholds = QualityThresholds(
        stale_run_length=cfg.stale_run_length,
        max_abs_log_return=cfg.max_abs_log_return,
    )
    structural = lakehouse_quality_report(frame, thresholds)
    checks = translate_lakehouse_checks(structural, frame, symbol, clock, cfg)
    checks.extend(
        [
            check_non_finite(frame, cfg),
            check_timezone(frame, clock, cfg),
            check_volume(frame, cfg),
            check_mad_outliers(frame, symbol, clock, cfg),
            check_missing_bars(frame, symbol, clock, cfg),
        ]
    )

    applicable = [check for check in checks if check.applicable]
    weight_total = sum(check.weight for check in applicable)
    score = (
        1.0
        if weight_total <= 0
        else sum(check.weight * check.pass_rate for check in applicable) / weight_total
    )
    passed = all(check.violations == 0 for check in applicable)

    start, end = _span(frame, clock)
    return QualityReport(
        dataset=dataset,
        rows=frame.height,
        symbols=None if symbol is None else int(frame[symbol].n_unique()),
        symbol_column=symbol,
        clock_column=clock,
        start=start,
        end=end,
        passed=passed,
        score=score,
        checks=checks,
        thresholds={
            "stale_run_length": cfg.stale_run_length,
            "max_abs_log_return": cfg.max_abs_log_return,
            "mad_z_threshold": cfg.mad_z_threshold,
            "expected_interval_seconds": cfg.expected_interval_seconds,
            "gap_factor": cfg.gap_factor,
        },
    )


def score_datasets(
    frames: dict[str, Any],
    config: QualityConfig | None = None,
) -> dict[str, QualityReport]:
    """Score several named datasets; keys become ``QualityReport.dataset``."""
    return {name: score_bars(frame, config, dataset=name) for name, frame in frames.items()}


def _as_polars(df: Any) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    module = type(df).__module__.split(".")[0]
    if module == "pandas":
        return pl.from_pandas(df)
    raise TypeError(f"score_bars needs a polars or pandas DataFrame, got {type(df)!r}")


def _span(frame: pl.DataFrame, clock: str | None) -> tuple[str | None, str | None]:
    if clock is None or clock not in frame.columns or frame.height == 0:
        return None, None
    dtype = frame.schema[clock]
    if not isinstance(dtype, pl.Datetime | pl.Date) and not dtype.is_numeric():
        return None, None
    start = frame[clock].min()
    end = frame[clock].max()
    return (None if start is None else str(start), None if end is None else str(end))
