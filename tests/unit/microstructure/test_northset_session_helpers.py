"""Unit tests for session-candle helpers extracted for the McCabe ratchet."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from quant_fund.northset.identities import (
    _empty_session_schema,
    _session_rows_for_daily_bar,
    session_candles_from_daily,
)


def test_empty_session_schema_keys() -> None:
    schema = _empty_session_schema()
    assert "security_id" in schema and "session_index" in schema


def test_session_rows_skip_non_finite_and_nonpositive_close() -> None:
    parent = datetime(2024, 1, 2, tzinfo=UTC)
    bad = {
        "security_id": "A",
        "event_time": parent,
        "open": float("nan"),
        "high": 1.0,
        "low": 1.0,
        "close": 1.0,
    }
    assert (
        _session_rows_for_daily_bar(
            bad,
            n_candles=4,
            high_first=True,
            has_symbol=False,
            has_available=False,
            has_volume=False,
        )
        == []
    )
    nonpos = {
        "security_id": "A",
        "event_time": parent,
        "open": 1.0,
        "high": 1.1,
        "low": 0.9,
        "close": 0.0,
    }
    assert (
        _session_rows_for_daily_bar(
            nonpos,
            n_candles=4,
            high_first=False,
            has_symbol=False,
            has_available=False,
            has_volume=False,
        )
        == []
    )


def test_session_rows_expand_valid_bar() -> None:
    parent = datetime(2024, 1, 2, tzinfo=UTC)
    row = {
        "security_id": "A",
        "event_time": parent,
        "available_time": parent,
        "open": 100.0,
        "high": 110.0,
        "low": 90.0,
        "close": 105.0,
        "volume": 8.0,
        "symbol": "AAA",
    }
    rows = _session_rows_for_daily_bar(
        row,
        n_candles=4,
        high_first=True,
        has_symbol=True,
        has_available=True,
        has_volume=True,
    )
    assert len(rows) == 4
    assert rows[0]["open"] == pytest.approx(100.0)
    assert rows[-1]["close"] == pytest.approx(105.0)
    assert rows[0]["volume"] == pytest.approx(2.0)
    assert rows[0]["symbol"] == "AAA"


def test_session_candles_from_daily_empty() -> None:
    empty = pl.DataFrame(
        schema={
            "security_id": pl.String,
            "event_time": pl.Datetime(time_zone="UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
        }
    )
    out = session_candles_from_daily(empty, n_candles=4, seed=1)
    assert out.height == 0
