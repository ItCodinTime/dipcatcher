"""Lookahead fuzzer: future prices and future targets do not move past fills."""

from __future__ import annotations

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from quant_fund.backtest.engine import run_backtest
from quant_fund.data.point_in_time import filter_available, validate_feature_frame
from quant_fund.data.sources.base import pit_frame
from tests.property._books import T0, daily_bars, research_config, weights_frame
from tests.property._profiles import adversarial_settings


def _panel(n_days: int = 6) -> tuple[pl.DataFrame, pl.DataFrame]:
    prices = []
    px_a, px_b = 100.0, 80.0
    for day in range(n_days):
        px_a *= 1.0 + 0.001 * ((day % 5) - 2)
        px_b *= 1.0 - 0.0005 * ((day % 3) - 1)
        prices.append([px_a, px_b])
    bars = daily_bars(prices)
    grid = weights_frame(n_days, 2, 0.3)
    return bars, grid


def _fills_through(result, cutoff_day: int) -> list[tuple[object, str, float, float]]:
    if result.fills.height == 0:
        return []
    cutoff = result.equity["event_time"][cutoff_day]
    # Equity rows are execution dates. Compare fill timestamps directly.
    kept = []
    for row in result.fills.iter_rows(named=True):
        if row["fill_time"] <= cutoff:
            kept.append(
                (
                    row["fill_time"],
                    row["security_id"],
                    float(row["quantity"]),
                    float(row["price"]),
                )
            )
    return kept


@given(
    shock=st.floats(min_value=0.5, max_value=1.8, allow_nan=False, allow_infinity=False),
    cutoff=st.integers(min_value=0, max_value=2),
)
@adversarial_settings()
def test_future_prices_do_not_change_past_fills(shock: float, cutoff: int) -> None:
    bars, grid = _panel()
    cfg = research_config(commission_bps=2.0, half_spread_bps=1.0)
    base = run_backtest(bars, grid, cfg, initial_nav=1_000_000.0)
    future = bars.with_columns(
        pl.when(pl.col("event_time") > base.equity["event_time"][cutoff])
        .then(pl.col("open") * shock)
        .otherwise(pl.col("open"))
        .alias("open"),
        pl.when(pl.col("event_time") > base.equity["event_time"][cutoff])
        .then(pl.col("close") * shock)
        .otherwise(pl.col("close"))
        .alias("close"),
        pl.when(pl.col("event_time") > base.equity["event_time"][cutoff])
        .then(pl.col("close_total_return") * shock)
        .otherwise(pl.col("close_total_return"))
        .alias("close_total_return"),
        pl.when(pl.col("event_time") > base.equity["event_time"][cutoff])
        .then(pl.col("adv") * shock)
        .otherwise(pl.col("adv"))
        .alias("adv"),
    )
    shocked = run_backtest(future, grid, cfg, initial_nav=1_000_000.0)
    assert _fills_through(base, cutoff) == _fills_through(shocked, cutoff)
    assert base.equity["nav"].to_list()[: cutoff + 1] == pytest.approx(
        shocked.equity["nav"].to_list()[: cutoff + 1]
    )


@given(cutoff=st.integers(min_value=0, max_value=3))
@adversarial_settings()
def test_future_targets_do_not_change_earlier_fills(cutoff: int) -> None:
    bars, grid = _panel()
    cfg = research_config(frictionless=True)
    base = run_backtest(bars, grid, cfg, initial_nav=1_000_000.0)
    cutoff_time = bars["event_time"].unique().sort()[cutoff]
    shocked_w = grid.with_columns(
        pl.when(pl.col("event_time") > cutoff_time)
        .then(pl.lit(-0.1))
        .otherwise(pl.col("target_weight"))
        .alias("target_weight")
    )
    shocked = run_backtest(bars, shocked_w, cfg, initial_nav=1_000_000.0)

    # A target dated after the cutoff cannot change a fill whose signal is
    # at or before the cutoff. Next-open fills use the signal day's target.
    def _signaled(result) -> list[tuple[object, str, float]]:
        if result.fills.height == 0:
            return []
        return [
            (row["signal_time"], row["security_id"], float(row["quantity"]))
            for row in result.fills.iter_rows(named=True)
            if row["signal_time"] <= cutoff_time
        ]

    assert _signaled(base) == _signaled(shocked)


@given(
    lag_hours=st.integers(min_value=0, max_value=48),
    future_hours=st.integers(min_value=1, max_value=72),
)
@adversarial_settings()
def test_pit_filter_drops_unpublished_rows(lag_hours: int, future_hours: int) -> None:
    from datetime import timedelta

    event = T0
    available = T0 + timedelta(hours=lag_hours)
    ingested = T0 + timedelta(days=5)
    decision = available
    frame = pit_frame(
        [
            {
                "security_id": "A",
                "symbol": "A",
                "event_time": event,
                "available_time": available,
                "value": 1.0,
            },
            {
                "security_id": "B",
                "symbol": "B",
                "event_time": event,
                "available_time": decision + timedelta(hours=future_hours),
                "value": 2.0,
            },
        ],
        source="fixture",
    )
    # pit_frame stamps ingested_time as now, which is after these fixtures.
    assert frame["ingested_time"].max() >= frame["available_time"].max()
    visible = filter_available(frame, decision)
    assert visible["security_id"].to_list() == ["A"]
    validate_feature_frame(visible, decision)
    assert ingested > available
