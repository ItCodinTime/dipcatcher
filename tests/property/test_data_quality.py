"""Property tests for quant_fund.data.quality — SYNTHETIC frames only.

Three invariants: clean synthetic bars score clean, planted faults are
always detected by the owning check, and reports are deterministic.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.data.quality import report_json, score_bars

BASE = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)

_RETURNS = st.one_of(
    st.floats(-0.05, -1e-6, allow_nan=False, allow_infinity=False),
    st.floats(1e-6, 0.05, allow_nan=False, allow_infinity=False),
)


def _clean_frame(
    n: int, base: float, ret: float, interval_seconds: int, vol_base: float
) -> pl.DataFrame:
    """Synthetic bars engineered clean: geometric close, regular spacing."""
    close = [base * math.exp(ret * i) for i in range(n)]
    return pl.DataFrame(
        {
            "symbol": ["SYN"] * n,
            "event_time": [BASE + timedelta(seconds=interval_seconds * i) for i in range(n)],
            "open": [c * 0.999 for c in close],
            "high": [c * 1.01 for c in close],
            "low": [c * 0.99 for c in close],
            "close": close,
            "volume": [vol_base + i for i in range(n)],
        }
    )


_CLEAN_ARGS = {
    "n": st.integers(12, 80),
    "base": st.floats(0.5, 1e4, allow_nan=False, allow_infinity=False),
    "ret": _RETURNS,
    "interval": st.sampled_from([60, 300, 3600, 86400]),
    "vol": st.floats(1.0, 1e7, allow_nan=False, allow_infinity=False),
}


@given(**_CLEAN_ARGS)
@settings(max_examples=40, deadline=None)
def test_clean_synthetic_bars_score_clean(
    n: int, base: float, ret: float, interval: int, vol: float
) -> None:
    frame = _clean_frame(n, base, ret, interval, vol)
    report = score_bars(frame)
    assert report.passed is True
    assert report.score == 1.0
    assert all(check.violations == 0 for check in report.checks)


_FAULTS = (
    "duplicate",
    "shuffle",
    "ohlc_break",
    "bad_volume",
    "nan_close",
    "stale_run",
    "drop_bars",
    "naive_tz",
    "spike",
)
_FAULT_CHECK = {
    "duplicate": "duplicates",
    "shuffle": "non_monotone",
    "ohlc_break": "ohlc",
    "bad_volume": "volume",
    "nan_close": "non_finite",
    "stale_run": "stale_prices",
    "drop_bars": "missing_bars",
    "naive_tz": "timezone",
    "spike": "mad_outliers",
}


def _apply_fault(frame: pl.DataFrame, fault: str, i: int, j: int) -> pl.DataFrame:
    n = frame.height
    when = pl.arange(0, n)
    if fault == "duplicate":
        return pl.concat([frame, frame.slice(i, 1)])
    if fault == "shuffle":
        rows = frame.to_dicts()
        rows[i], rows[j] = rows[j], rows[i]
        return pl.DataFrame(rows)
    if fault == "ohlc_break":
        return frame.with_columns(
            pl.when(when == i).then(pl.col("low") * 0.5).otherwise(pl.col("high")).alias("high")
        )
    if fault == "bad_volume":
        return frame.with_columns(
            pl.when(when == i).then(0.0).otherwise(pl.col("volume")).alias("volume")
        )
    if fault == "nan_close":
        return frame.with_columns(
            pl.when(when == i).then(float("nan")).otherwise(pl.col("close")).alias("close")
        )
    if fault == "stale_run":
        return frame.with_columns(
            pl.when(when.is_between(i, min(i + 6, n - 1)))
            .then(7.5)
            .otherwise(pl.col("close"))
            .alias("close")
        )
    if fault == "drop_bars":
        return frame.with_row_index().filter(pl.col("index") != i).drop("index")
    if fault == "naive_tz":
        return frame.with_columns(pl.col("event_time").dt.replace_time_zone(None))
    if fault == "spike":
        return frame.with_columns(
            pl.when(when == i)
            .then(pl.col("close") * 50.0)
            .otherwise(pl.col("close"))
            .alias("close")
        )
    raise AssertionError(fault)


@given(
    **_CLEAN_ARGS,
    fault=st.sampled_from(_FAULTS),
    i=st.integers(1, 70),
    j=st.integers(1, 70),
)
@settings(max_examples=60, deadline=None)
def test_planted_faults_are_always_detected(
    n: int,
    base: float,
    ret: float,
    interval: int,
    vol: float,
    fault: str,
    i: int,
    j: int,
) -> None:
    frame = _clean_frame(n, base, ret, interval, vol)
    i, j = min(i, j) % n, max(i, j) % n
    if fault == "shuffle" and i == j:
        j = (j + 1) % n
        i, j = min(i, j), max(i, j)
    if fault in {"nan_close", "spike"}:
        i = max(1, i)  # need a previous close to form a return
    if fault == "stale_run":
        i = min(i, n - 5)  # leave room for a >= 5-bar identical-close run
    if fault == "drop_bars":
        i = 1 + i % (n - 2)  # only interior drops create a spacing gap
    dirty = _apply_fault(frame, fault, i, j)
    report = score_bars(dirty)
    check = report.check(_FAULT_CHECK[fault])
    assert check is not None
    assert check.violations > 0, f"{fault} not detected: {check.detail}"
    assert report.passed is False
    assert 0.0 <= report.score < 1.0


@given(**_CLEAN_ARGS, fault=st.sampled_from(_FAULTS), i=st.integers(1, 70))
@settings(max_examples=40, deadline=None)
def test_reports_are_deterministic(
    n: int, base: float, ret: float, interval: int, vol: float, fault: str, i: int
) -> None:
    frame = _apply_fault(
        _clean_frame(n, base, ret, interval, vol), fault, max(1, i % n), (i + 3) % n
    )
    first = score_bars(frame, dataset="det")
    second = score_bars(frame, dataset="det")
    assert report_json(first) == report_json(second)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


@given(**_CLEAN_ARGS, fault=st.sampled_from(_FAULTS), i=st.integers(1, 70))
@settings(max_examples=40, deadline=None)
def test_scorecard_invariants(
    n: int, base: float, ret: float, interval: int, vol: float, fault: str, i: int
) -> None:
    frame = _apply_fault(_clean_frame(n, base, ret, interval, vol), fault, i % n, (i + 5) % n)
    report = score_bars(frame)
    assert 0.0 <= report.score <= 1.0
    for check in report.checks:
        assert check.violations >= 0
        assert check.checked >= 0
        assert len(check.worst) <= 8
    if report.passed:
        assert report.score == 1.0
