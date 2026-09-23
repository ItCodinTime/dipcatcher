"""Regression checks for full target snapshots and causal sleeve allocation."""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from quant_fund.backtest.adaptive_mix import (
    banded_targets,
    dense_targets,
    mix_targets,
    trailing_nav_allocations,
)


def _dates(n: int) -> list[datetime]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return [start + timedelta(days=i) for i in range(n)]


def test_missing_proposal_emits_explicit_zero_exit() -> None:
    dates = _dates(3)
    bars = pl.DataFrame({"event_time": dates, "security_id": ["BTC"] * 3})
    sparse = pl.DataFrame(
        {"event_time": [dates[0]], "security_id": ["BTC"], "target_weight": [0.2]}
    )
    dense = dense_targets(bars, sparse)
    assert dense["target_weight"].to_list() == [0.2, 0.0, 0.0]
    allocation = pl.DataFrame({"event_time": dates, "first": [0.5] * 3, "second": [0.5] * 3})
    flat = dense.with_columns(pl.lit(0.0).alias("target_weight"))
    mixed = mix_targets({"first": dense, "second": flat}, allocation)
    assert mixed["target_weight"].to_list() == [0.1, 0.0, 0.0]


def test_future_nav_change_cannot_change_earlier_allocation() -> None:
    dates = _dates(12)
    nav_a = np.cumprod(np.r_[1.0, [1.01, 0.995] * 5 + [1.01]]) * 100.0
    nav_b = np.cumprod(np.r_[1.0, [0.995, 1.01] * 5 + [0.995]]) * 100.0
    a = pl.DataFrame({"event_time": dates, "nav": nav_a})
    b = pl.DataFrame({"event_time": dates, "nav": nav_b})
    base = trailing_nav_allocations({"a": a, "b": b}, window=5, min_obs=2)
    changed = a.with_columns(
        pl.when(pl.col("event_time") == dates[-1])
        .then(pl.col("nav") * 2.0)
        .otherwise(pl.col("nav"))
        .alias("nav")
    )
    rerun = trailing_nav_allocations({"a": changed, "b": b}, window=5, min_obs=2)
    assert base.head(11).equals(rerun.head(11))
    assert base.tail(1)["a"][0] != rerun.tail(1)["a"][0]
    assert all(abs(x - 1.0) < 1e-12 for x in base.select(pl.sum_horizontal("a", "b")).to_series())


def test_allocator_rejects_mismatched_calendar() -> None:
    dates = _dates(3)
    a = pl.DataFrame({"event_time": dates, "nav": [100.0, 101.0, 102.0]})
    b = pl.DataFrame({"event_time": dates[:2], "nav": [100.0, 99.0]})
    with pytest.raises(ValueError, match="calendar mismatch"):
        trailing_nav_allocations({"a": a, "b": b}, window=2, min_obs=2)


def test_band_suppresses_tiny_revisions_but_keeps_exit() -> None:
    dates = _dates(5)
    targets = pl.DataFrame(
        {
            "event_time": dates,
            "security_id": ["BTC"] * 5,
            "target_weight": [0.2, 0.205, 0.21, 0.22, 0.0],
        }
    )
    filtered = banded_targets(targets, band=0.01)
    assert filtered["target_weight"].to_list() == [0.2, 0.22, 0.0]
    assert filtered["event_time"].to_list() == [dates[0], dates[3], dates[4]]
