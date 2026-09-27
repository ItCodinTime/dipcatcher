"""Canonical normalized output schemas shared by every engine adapter.

Each adapter emits the same three long-format polars frames so reconciliation
never touches engine-native objects:

- ``trades``: one row per executed fill
    (engine, strategy, security_id, fill_time, side, quantity, price, fee)
- ``positions``: one row per (event_time, security_id) holding a nonzero share
    balance, derived uniformly as the cumulative sum of fill quantities plus
    shares * close mark
    (engine, strategy, security_id, event_time, quantity, market_value)
- ``equity``: one row per bar
    (engine, strategy, event_time, nav, cash)

``side`` is +1.0 for buys and -1.0 for sells. ``quantity`` is always positive
(direction lives in ``side``) unless the engine natively reports signed fills,
in which case adapters normalize to positive quantity + signed side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

TRADES_SCHEMA: dict[str, pl.DataType] = {
    "engine": pl.String,
    "strategy": pl.String,
    "security_id": pl.String,
    "fill_time": pl.Datetime("us", "UTC"),
    "side": pl.Float64,
    "quantity": pl.Float64,
    "price": pl.Float64,
    "fee": pl.Float64,
}

POSITIONS_SCHEMA: dict[str, pl.DataType] = {
    "engine": pl.String,
    "strategy": pl.String,
    "security_id": pl.String,
    "event_time": pl.Datetime("us", "UTC"),
    "quantity": pl.Float64,
    "market_value": pl.Float64,
}

EQUITY_SCHEMA: dict[str, pl.DataType] = {
    "engine": pl.String,
    "strategy": pl.String,
    "event_time": pl.Datetime("us", "UTC"),
    "nav": pl.Float64,
    "cash": pl.Float64,
}


def empty_trades() -> pl.DataFrame:
    return pl.DataFrame(schema=TRADES_SCHEMA)


def empty_positions() -> pl.DataFrame:
    return pl.DataFrame(schema=POSITIONS_SCHEMA)


def empty_equity() -> pl.DataFrame:
    return pl.DataFrame(schema=EQUITY_SCHEMA)


@dataclass
class EngineRun:
    """One (engine, strategy) backtest result in canonical form."""

    engine: str
    strategy: str
    status: str  # "ok" | "skip" | "fail"
    trades: pl.DataFrame = field(default_factory=empty_trades)
    positions: pl.DataFrame = field(default_factory=empty_positions)
    equity: pl.DataFrame = field(default_factory=empty_equity)
    notes: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def skip(cls, engine: str, strategy: str, reason: str) -> EngineRun:
        return cls(engine=engine, strategy=strategy, status="skip", notes=[reason])

    @classmethod
    def fail(cls, engine: str, strategy: str, reason: str) -> EngineRun:
        return cls(engine=engine, strategy=strategy, status="fail", notes=[reason])


def positions_from_trades(
    trades: pl.DataFrame,
    close_marks: pl.DataFrame,
    *,
    engine: str,
    strategy: str,
) -> pl.DataFrame:
    """Derive a positions frame from fills + close marks.

    ``close_marks`` columns: (event_time, security_id, close). Share balances
    are the cumulative signed fill quantity; market_value uses the close mark
    on each date (forward-filled by construction of the panel — adapters pass
    the full panel so every traded name has a mark on every later date).
    """
    if trades.height == 0:
        return empty_positions()
    signed = trades.with_columns(
        (pl.col("quantity") * pl.col("side")).alias("signed_qty"),
    )
    closes = close_marks.select("event_time", "security_id", "close").sort(
        "security_id", "event_time"
    )
    dates = closes.select("event_time").unique().sort("event_time")
    sids = signed.select("security_id").unique()
    grid = dates.join(sids, how="cross")
    fills = signed.group_by(["security_id", "fill_time"]).agg(
        pl.col("signed_qty").sum().alias("qty_delta")
    )
    daily = (
        grid.join(
            fills,
            left_on=["security_id", "event_time"],
            right_on=["security_id", "fill_time"],
            how="left",
        )
        .with_columns(pl.col("qty_delta").fill_null(0.0))
        .sort("security_id", "event_time")
        .with_columns(
            pl.col("qty_delta").cum_sum().over("security_id").alias("quantity")
        )
        .join(closes, on=["security_id", "event_time"], how="left")
        .with_columns((pl.col("quantity") * pl.col("close")).alias("market_value"))
        .filter(pl.col("quantity").abs() > 1e-12)
        .select(
            pl.lit(engine).alias("engine"),
            pl.lit(strategy).alias("strategy"),
            "security_id",
            "event_time",
            "quantity",
            "market_value",
        )
    )
    return daily.cast(POSITIONS_SCHEMA)


def marks_from_bars(bars: pl.DataFrame) -> pl.DataFrame:
    """(event_time, security_id, close) mark frame used for position valuation."""
    return bars.select("event_time", "security_id", "close").sort(
        "security_id", "event_time"
    )
