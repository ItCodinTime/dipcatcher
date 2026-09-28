"""Unit tests for quant_fund.data.quality — per-check, on synthetic frames.

All frames are SYNTHETIC; nothing here is market evidence.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from quant_fund.data.quality import (
    QUALITY_REPORT_SCHEMA,
    QualityConfig,
    QualityReport,
    report_json,
    report_sha256,
    score_bars,
    score_datasets,
    write_report,
)
from quant_fund.data.quality.cli import main as cli_main

BASE = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
N = 40


def _clean(n: int = N, *, symbol: str = "AAPL") -> pl.DataFrame:
    """Regularly spaced, tz-aware, internally consistent synthetic bars."""
    close = [10.0 * math.exp(0.001 * i) for i in range(n)]
    return pl.DataFrame(
        {
            "symbol": [symbol] * n,
            "event_time": [BASE + timedelta(minutes=i) for i in range(n)],
            "open": [c * 0.999 for c in close],
            "high": [c * 1.01 for c in close],
            "low": [c * 0.99 for c in close],
            "close": close,
            "volume": [100.0 + i for i in range(n)],
        }
    )


def _with_close(frame: pl.DataFrame, index: int, value: float) -> pl.DataFrame:
    n = frame.height
    return frame.with_columns(
        pl.when(pl.arange(0, n) == index).then(value).otherwise(pl.col("close")).alias("close")
    )


def _with_volume(frame: pl.DataFrame, index: int, value: float) -> pl.DataFrame:
    n = frame.height
    return frame.with_columns(
        pl.when(pl.arange(0, n) == index).then(value).otherwise(pl.col("volume")).alias("volume")
    )


def test_clean_frame_passes_at_perfect_score() -> None:
    report = score_bars(_clean(), dataset="clean")
    assert report.schema_ == QUALITY_REPORT_SCHEMA
    assert report.dataset == "clean"
    assert report.rows == N
    assert report.symbols == 1
    assert report.passed is True
    assert report.score == 1.0
    for check in report.checks:
        assert check.applicable is True, check.name
        assert check.violations == 0, check.name


def test_duplicate_timestamps_detected() -> None:
    frame = pl.concat([_clean(), _clean().head(2)])
    report = score_bars(frame)
    check = report.check("duplicates")
    # lakehouse counts every row participating in a duplicated key (4 rows
    # across 2 duplicated timestamps)
    assert check is not None and check.violations == 4
    assert report.passed is False
    assert all(o.timestamp is not None for o in check.worst)


def test_non_monotone_timestamps_detected() -> None:
    frame = _clean()
    rows = frame.to_dicts()
    rows[5], rows[20] = rows[20], rows[5]
    reordered = pl.DataFrame(rows)
    report = score_bars(reordered)
    check = report.check("non_monotone")
    assert check is not None and check.violations > 0


def test_ohlc_violations_detected() -> None:
    frame = _clean()
    n = frame.height
    broken = frame.with_columns(
        pl.when(pl.arange(0, n) == 7).then(5.0).otherwise(pl.col("high")).alias("high")
    )  # high < low
    report = score_bars(broken)
    check = report.check("ohlc")
    assert check is not None and check.violations == 1


def test_zero_and_negative_volume_detected() -> None:
    frame = _with_volume(_with_volume(_clean(), 3, 0.0), 9, -5.0)
    check = score_bars(frame).check("volume")
    assert check is not None and check.violations == 2
    assert check.checked == N


def test_mad_outlier_detected_at_spike() -> None:
    frame = _with_close(_clean(), 20, 50.0)
    check = score_bars(frame).check("mad_outliers")
    assert check is not None and check.violations >= 1
    assert check.worst[0].value is not None
    assert check.worst[0].timestamp is not None


def test_mad_outlier_ignores_smooth_paths() -> None:
    # Geometric close path: constant log return, zero outlier by design.
    report = score_bars(_clean())
    check = report.check("mad_outliers")
    assert check is not None and check.violations == 0


def test_stale_run_detected() -> None:
    frame = _clean()
    n = frame.height
    stale = frame.with_columns(
        pl.when(pl.arange(0, n).is_between(5, 12))
        .then(7.5)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    check = score_bars(stale).check("stale_prices")
    assert check is not None and check.violations == 1


def test_missing_bars_detected_with_estimate() -> None:
    frame = _clean().with_row_index()
    gapped = frame.filter(~pl.col("index").is_in([10, 11])).drop("index")
    check = score_bars(gapped).check("missing_bars")
    assert check is not None and check.violations == 1
    assert check.worst[0].extra["estimated_missing"] == 2
    assert check.worst[0].timestamp is not None


def test_missing_bars_configured_interval() -> None:
    config = QualityConfig(expected_interval_seconds=60.0, gap_factor=1.5)
    frame = _clean().with_row_index()
    gapped = frame.filter(pl.col("index") != 10).drop("index")
    check = score_bars(gapped, config).check("missing_bars")
    assert check is not None and check.violations == 1


def test_non_finite_values_detected() -> None:
    frame = _clean()
    n = frame.height
    dirty = frame.with_columns(
        pl.when(pl.arange(0, n) == 3).then(float("nan")).otherwise(pl.col("close")).alias("close"),
        pl.when(pl.arange(0, n) == 8).then(float("inf")).otherwise(pl.col("high")).alias("high"),
    )
    check = score_bars(dirty).check("non_finite")
    assert check is not None and check.violations == 2
    assert "close" in check.detail and "high" in check.detail


def test_null_cells_count_as_non_finite() -> None:
    frame = _clean().with_columns(
        pl.when(pl.arange(0, N) == 4).then(None).otherwise(pl.col("volume")).alias("volume")
    )
    check = score_bars(frame).check("non_finite")
    assert check is not None and check.violations == 1


def test_timezone_naive_clock_detected() -> None:
    naive = _clean().with_columns(pl.col("event_time").dt.replace_time_zone(None))
    report = score_bars(naive)
    check = report.check("timezone")
    assert check is not None and check.violations == N
    assert report.passed is False
    # everything else is still clean, so the score is below 1 but not zero
    assert 0.0 < report.score < 1.0


def test_multi_symbol_frames_score_per_symbol() -> None:
    aapl = _clean()
    msft = _clean().with_columns(pl.lit("MSFT").alias("symbol"))
    frame = pl.concat([aapl, msft])
    report = score_bars(frame)
    assert report.symbols == 2
    assert report.passed is True
    # gaps do not leak across the symbol boundary
    assert report.check("missing_bars").violations == 0


def test_no_symbol_column_still_scores() -> None:
    frame = _clean().drop("symbol")
    report = score_bars(frame)
    assert report.symbols is None
    assert report.check("duplicates").applicable is False
    assert report.check("missing_bars").applicable is True
    assert report.passed is True


def test_pandas_input_accepted() -> None:
    pytest.importorskip("pandas")
    frame = _clean().to_pandas()
    report = score_bars(frame)
    assert report.rows == N
    assert report.passed is True


def test_rejects_unsupported_frame_type() -> None:
    with pytest.raises(TypeError):
        score_bars({"close": [1.0]})


def test_report_json_round_trip(tmp_path: Path) -> None:
    report = score_bars(_clean(), dataset="rt")
    text = report_json(report)
    parsed = json.loads(text)
    assert parsed["schema"] == QUALITY_REPORT_SCHEMA
    assert parsed["score"] == 1.0
    revived = QualityReport.model_validate(parsed)
    assert revived == report
    out = write_report(report, tmp_path / "q" / "report.json")
    assert QualityReport.model_validate(json.loads(out.read_text())) == report
    assert report_sha256(report) == report_sha256(revived)


def test_report_is_deterministic() -> None:
    assert report_json(score_bars(_clean())) == report_json(score_bars(_clean()))


def test_score_datasets_names_reports() -> None:
    reports = score_datasets({"a": _clean(), "b": _with_close(_clean(), 1, 99.0)})
    assert set(reports) == {"a", "b"}
    assert reports["a"].dataset == "a" and reports["a"].passed
    assert reports["b"].passed is False


def test_config_overrides_change_results() -> None:
    frame = _clean()
    n = frame.height
    short_stale = frame.with_columns(
        pl.when(pl.arange(0, n).is_between(5, 8))
        .then(7.5)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    assert score_bars(short_stale).check("stale_prices").violations == 0
    lax = QualityConfig(stale_run_length=3)
    assert score_bars(short_stale, lax).check("stale_prices").violations == 1


def test_cli_scores_parquet(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "bars.parquet"
    _clean().write_parquet(path)
    assert cli_main([str(path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["passed"] is True

    dirty = _with_close(_clean(), 5, 99.0)
    dirty.write_parquet(path)
    out_path = tmp_path / "report.json"
    assert cli_main([str(path), "--out", str(out_path)]) == 1
    written = json.loads(out_path.read_text())
    assert written["passed"] is False


def test_empty_frame_scores_vacuously() -> None:
    report = score_bars(_clean().head(0))
    assert report.rows == 0
    assert report.passed is True
    assert report.score == 1.0
