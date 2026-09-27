"""Duplicate prints must not invent regular-session coverage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from quant_fund.data.adapters.hf_ohlcv_1m import minute_gap_report


def _row(minute: int, *, session: str = "rth") -> dict[str, object]:
    # 2024-01-02 14:30 UTC is 09:30 America/New_York.
    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC) + timedelta(minutes=minute)
    event = start + timedelta(minutes=1)
    return {
        "security_id": "A",
        "symbol": "A",
        "event_time": event,
        "available_time": event,
        "ingested_time": datetime(2026, 1, 1, tzinfo=UTC),
        "source": "fixture",
        "revision_id": "v",
        "open": 10.0,
        "high": 11.0,
        "low": 9.5,
        "close": 10.5,
        "volume": 1.0,
        "currency": "USD",
        "session": session,
    }


def _frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(
        pl.col("event_time").cast(pl.Datetime("us", "UTC")),
        pl.col("available_time").cast(pl.Datetime("us", "UTC")),
        pl.col("ingested_time").cast(pl.Datetime("us", "UTC")),
    )


def test_duplicate_rth_minutes_do_not_reduce_the_gap() -> None:
    rows = [_row(i) for i in range(10)]
    rows.extend(_row(0) for _ in range(400))
    report = minute_gap_report(_frame(rows))
    assert report.height == 1
    assert int(report["n_rth"][0]) == 10
    assert int(report["n_rth_expected"][0]) == 390
    assert int(report["n_rth_missing"][0]) == 380
    assert int(report["n_rth"][0]) + int(report["n_rth_missing"][0]) == 390


def test_off_clock_rows_labeled_rth_are_not_regular_session_minutes() -> None:
    # 00:00 ET through the next few hours, mislabeled as rth. None of these
    # minute starts sit in 09:30–16:00, so the regular session is fully missing.
    rows = []
    for minute in range(400):
        start = datetime(2024, 1, 2, 5, 0, tzinfo=UTC) + timedelta(minutes=minute)
        event = start + timedelta(minutes=1)
        row = _row(0)
        row["event_time"] = event
        row["available_time"] = event
        rows.append(row)
    report = minute_gap_report(_frame(rows))
    assert int(report["n_rth"][0]) == 0
    assert int(report["n_rth_missing"][0]) == 390
    assert int(report["n_rth_missing"][0]) >= 0
