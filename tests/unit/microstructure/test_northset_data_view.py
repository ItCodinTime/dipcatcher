"""Corporate-action-safe Northset market-view tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_fund.northset.data_view import canonical_northset_bars
from quant_fund.northset.sweeps import liquidity_sweep_frame


def _split_fixture() -> pl.DataFrame:
    n = 30
    split_index = 20
    event_time = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]
    raw = [100.0] * split_index + [50.0] * (n - split_index)
    factor = [2.0] * split_index + [1.0] * (n - split_index)
    return pl.DataFrame(
        {
            "security_id": ["A"] * n,
            "event_time": event_time,
            "available_time": event_time,
            "open": raw,
            "high": [x * 1.01 for x in raw],
            "low": [x * 0.99 for x in raw],
            "close": raw,
            "volume": [1_000.0] * split_index + [2_000.0] * (n - split_index),
            "split_factor": factor,
            "open_split_adjusted": [x / f for x, f in zip(raw, factor, strict=True)],
            "high_split_adjusted": [x * 1.01 / f for x, f in zip(raw, factor, strict=True)],
            "low_split_adjusted": [x * 0.99 / f for x, f in zip(raw, factor, strict=True)],
            "close_split_adjusted": [x / f for x, f in zip(raw, factor, strict=True)],
            "close_total_return": [50.0] * n,
        }
    )


def test_split_adjusted_view_removes_false_split_sweep() -> None:
    raw = _split_fixture()
    raw_sweeps = liquidity_sweep_frame(raw, lookback=10)
    assert raw_sweeps["sweep_low"][20] == 1.0
    view = canonical_northset_bars(raw)
    adjusted_sweeps = liquidity_sweep_frame(view.frame, lookback=10)
    assert adjusted_sweeps["sweep_low"][20] == 0.0
    assert view.price_basis == "split_adjusted"
    assert view.return_basis == "total_return"
    assert view.frame["volume"][0] == 2_000.0


def test_adjusted_view_fails_closed_on_missing_or_partial_columns() -> None:
    raw = _split_fixture().drop(
        [
            "open_split_adjusted",
            "high_split_adjusted",
            "low_split_adjusted",
            "close_split_adjusted",
            "close_total_return",
        ]
    )
    with pytest.raises(ValueError, match="requires split-adjusted"):
        canonical_northset_bars(raw)
    partial = raw.with_columns(pl.col("open").alias("open_split_adjusted"))
    with pytest.raises(ValueError, match="partial split-adjusted"):
        canonical_northset_bars(partial)
    fixture = canonical_northset_bars(raw, require_adjusted=False)
    assert fixture.price_basis == "raw_fixture_opt_out"
