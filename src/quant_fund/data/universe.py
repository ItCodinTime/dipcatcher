"""Point-in-time universe membership. Never use future ADV or future index lists."""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

import polars as pl

from quant_fund.config.models import UniverseConfig
from quant_fund.schemas.errors import LeakageError


def _visible_bars(bars: pl.DataFrame, when: datetime) -> pl.DataFrame:
    """Return bars observable by an as-of decision.

    Event time alone is insufficient for late-arriving or revised bars. When
    the optional availability timestamp is present, null or future availability
    is excluded rather than treated as observable.
    """
    visible = bars.filter(pl.col("event_time") <= when)
    if "available_time" in visible.columns:
        visible = visible.filter(pl.col("available_time") <= when)
    return visible


def trailing_adv(bars: pl.DataFrame, lookback: int) -> pl.DataFrame:
    dollar = pl.col("close") * pl.col("volume")
    return bars.sort(["security_id", "event_time"]).with_columns(
        dollar.rolling_mean(lookback).over("security_id").alias("adv"),
        pl.col("close").alias("px"),
        pl.len().over("security_id").alias("history_len_tmp"),
        pl.int_range(pl.len()).over("security_id").alias("bars_seen"),
    )


def membership_asof(
    bars: pl.DataFrame,
    master: pl.DataFrame,
    when: datetime,
    config: UniverseConfig,
) -> pl.DataFrame:
    """Eligible names using only bars observable at ``when``."""
    hist = _visible_bars(bars, when)
    if hist.is_empty():
        return hist.clear()
    enriched = trailing_adv(hist, lookback=20)
    last = (
        enriched.sort("event_time")
        .group_by("security_id")
        .agg(
            pl.all().last(),
            pl.len().alias("n_bars"),
        )
    )
    last = last.join(
        master.select(["security_id", "exchange", "sector", "industry", "security_type", "ticker"]),
        on="security_id",
        how="left",
    )
    last = last.filter(
        (pl.col("px") >= config.min_price)
        & (pl.col("adv").fill_null(0.0) >= config.min_adv)
        & (pl.col("n_bars") >= config.min_history_bars)
        & (pl.col("exchange").is_in(config.exchanges))
        & (pl.col("security_type").is_in(config.security_types))
    )
    if config.top_n_adv is not None:
        last = last.sort("adv", descending=True).head(config.top_n_adv)
    return last.with_columns(
        pl.lit(when).alias("effective_from"),
        pl.col("ticker").alias("symbol"),
    )


def build_membership_panel(
    bars: pl.DataFrame,
    master: pl.DataFrame,
    timestamps: list[datetime],
    config: UniverseConfig,
) -> pl.DataFrame:
    frames = [
        membership_asof(bars, master, t, config).with_columns(pl.lit(t).alias("asof"))
        for t in timestamps
    ]
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def assert_universe_not_from_future(
    membership: pl.DataFrame, bars: pl.DataFrame, asof: datetime
) -> None:
    """Guard: membership at `asof` must not use bars after asof."""
    if "asof" in membership.columns:
        mem = membership.filter(pl.col("asof") == asof)
    else:
        mem = membership
    if mem.is_empty():
        return
    # Compare event-time history with the actually observable history. The
    # former intentionally retains late bars so this guard catches availability
    # leakage even when no future event-time bar changes the rolling window.
    leaked = trailing_adv(bars.filter(pl.col("event_time") <= asof), 20).filter(
        pl.col("event_time") == asof
    )
    pit = trailing_adv(_visible_bars(bars, asof), 20).filter(pl.col("event_time") == asof)
    if leaked.is_empty() or pit.is_empty():
        return
    joined = leaked.join(pit, on="security_id", suffix="_pit")
    if joined.is_empty():
        return
    max_diff = (joined["adv"] - joined["adv_pit"]).abs().max()
    if max_diff is not None and float(cast(Any, max_diff)) > 1e-6:
        # Using the full panel ADV at asof differs from PIT ADV: future leaked into rolling window
        raise LeakageError("Universe ADV used bars after asof")
    # Also verify the stamped membership ADV against the PIT recomputation:
    # catches a membership panel whose ADV was built from a full (leaky) panel.
    if {"security_id", "adv"} <= set(mem.columns):
        stamped = mem.select(["security_id", pl.col("adv").alias("adv_stamped")])
        check = stamped.join(
            pit.select(["security_id", pl.col("adv").alias("adv_pit")]),
            on="security_id",
            how="inner",
        )
        if check.height:
            stamped_diff = (check["adv_stamped"] - check["adv_pit"]).abs().max()
            if stamped_diff is not None and float(cast(Any, stamped_diff)) > 1e-6:
                raise LeakageError("Membership ADV does not match point-in-time ADV")
