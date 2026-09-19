"""Point-in-time universe membership. Never use future ADV or future index lists."""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

import polars as pl

from quant_fund.config.models import UniverseConfig
from quant_fund.data.corporate_actions import delisted_ids_asof, ticker_overrides_asof
from quant_fund.data.security_master import snapshot_asof
from quant_fund.schemas.errors import LeakageError, PointInTimeError

_MEMBERSHIP_FLAG = "_in_universe"


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


def require_valid_membership_panel(membership: pl.DataFrame) -> None:
    """Fail closed on a universe artifact that cannot be joined as PIT keys."""
    missing = [name for name in ("security_id", "asof") if name not in membership.columns]
    if missing:
        raise PointInTimeError(f"universe membership missing required columns: {missing}")
    blank_id = membership.filter(
        pl.col("security_id").is_null()
        | (pl.col("security_id").cast(pl.String).str.strip_chars() == "")
    )
    if blank_id.height:
        raise PointInTimeError("universe membership contains blank security_id")
    if membership.filter(pl.col("asof").is_null()).height:
        raise PointInTimeError("universe membership contains null asof")
    if membership.select(["security_id", "asof"]).is_duplicated().any():
        raise PointInTimeError("universe membership contains duplicate security_id/asof rows")


def membership_decision_keys(membership: pl.DataFrame) -> pl.DataFrame:
    """Unique ``(security_id, event_time)`` keys observable in the membership panel."""
    require_valid_membership_panel(membership)
    return (
        membership.select(
            pl.col("security_id").cast(pl.String),
            pl.col("asof").alias("event_time"),
        )
        .unique(maintain_order=True)
        .sort(["event_time", "security_id"])
    )


def attach_membership_flag(frame: pl.DataFrame, membership: pl.DataFrame) -> pl.DataFrame:
    """Stamp ``_in_universe`` by exact ``(security_id, event_time)`` membership.

    Name-level history is preserved. A later membership ``asof`` cannot mark an
    earlier bar eligible. Direct callers that omit membership keep the legacy
    unfiltered panel.
    """
    missing = [name for name in ("security_id", "event_time") if name not in frame.columns]
    if missing:
        raise PointInTimeError(f"panel missing membership join keys: {missing}")
    if _MEMBERSHIP_FLAG in frame.columns:
        frame = frame.drop(_MEMBERSHIP_FLAG)
    keys = membership_decision_keys(membership).with_columns(pl.lit(True).alias(_MEMBERSHIP_FLAG))
    out = frame.join(keys, on=["security_id", "event_time"], how="left")
    if _MEMBERSHIP_FLAG not in out.columns:
        return out.with_columns(pl.lit(False).alias(_MEMBERSHIP_FLAG))
    return out.with_columns(pl.col(_MEMBERSHIP_FLAG).fill_null(False))


def restrict_to_membership(frame: pl.DataFrame, membership: pl.DataFrame) -> pl.DataFrame:
    """Keep rows whose ``(security_id, event_time)`` is in the PIT universe.

    An empty membership panel cannot silently fall back to the unfiltered
    silver/gold frame: that would train and score names the universe rejected.
    """
    require_valid_membership_panel(membership)
    if frame.is_empty():
        return frame
    if membership.is_empty():
        raise PointInTimeError("universe membership is empty; refusing to use the unfiltered panel")
    flagged = attach_membership_flag(frame, membership)
    return flagged.filter(pl.col(_MEMBERSHIP_FLAG)).drop(_MEMBERSHIP_FLAG)


def require_panel_keys_in_membership(frame: pl.DataFrame, membership: pl.DataFrame) -> None:
    """Fail closed when a gold/decision panel contains non-member keys.

    Wave 108 inner-joins at materialization. Cached or hand-edited gold can
    later disagree with ``silver/universe.parquet``; training and forecast
    ``panel()`` must not silently keep ineligible names. Extra membership
    rows are allowed (incomplete gold is not leakage). Direct callers that
    never load membership keep the legacy unfiltered frame.
    """
    require_valid_membership_panel(membership)
    if frame.is_empty():
        return
    missing = [name for name in ("security_id", "event_time") if name not in frame.columns]
    if missing:
        raise PointInTimeError(f"panel missing membership join keys: {missing}")
    if membership.is_empty():
        raise PointInTimeError("universe membership is empty; refusing to use the unfiltered panel")
    extras = (
        frame.select(
            pl.col("security_id").cast(pl.String),
            "event_time",
        )
        .unique()
        .join(membership_decision_keys(membership), on=["security_id", "event_time"], how="anti")
    )
    if extras.height:
        raise PointInTimeError("gold/decision panel contains rows outside PIT universe membership")


def membership_asof(
    bars: pl.DataFrame,
    master: pl.DataFrame,
    when: datetime,
    config: UniverseConfig,
    actions: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Eligible names using only bars and listing events observable at ``when``."""
    hist = _visible_bars(bars, when)
    if hist.is_empty():
        return hist.clear()
    listing = actions if actions is not None else pl.DataFrame()
    dropped = set(delisted_ids_asof(listing, when, include_delisted=config.include_delisted))
    if dropped:
        hist = hist.filter(~pl.col("security_id").is_in(sorted(dropped)))
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
    snap = snapshot_asof(master, when)
    attr_cols = [
        name
        for name in (
            "security_id",
            "exchange",
            "sector",
            "industry",
            "security_type",
            "ticker",
        )
        if name in snap.columns
    ]
    if attr_cols:
        last = last.join(snap.select(attr_cols), on="security_id", how="left")
    last = last.filter(
        (pl.col("px") >= config.min_price)
        & (pl.col("adv").fill_null(0.0) >= config.min_adv)
        & (pl.col("n_bars") >= config.min_history_bars)
    )
    if "exchange" in last.columns:
        last = last.filter(pl.col("exchange").is_in(config.exchanges))
    if "security_type" in last.columns:
        last = last.filter(pl.col("security_type").is_in(config.security_types))
    if config.top_n_adv is not None:
        last = last.sort("adv", descending=True).head(config.top_n_adv)
    overrides = ticker_overrides_asof(listing, when)
    if not overrides.is_empty():
        last = last.join(overrides, on="security_id", how="left")
        if "ticker" in last.columns:
            last = last.with_columns(
                pl.coalesce(pl.col("new_ticker"), pl.col("ticker")).alias("ticker")
            ).drop("new_ticker")
        else:
            last = last.rename({"new_ticker": "ticker"})
    symbol = pl.col("ticker") if "ticker" in last.columns else pl.lit(None, dtype=pl.String)
    return last.with_columns(
        pl.lit(when).alias("effective_from"),
        symbol.alias("symbol"),
    )


def build_membership_panel(
    bars: pl.DataFrame,
    master: pl.DataFrame,
    timestamps: list[datetime],
    config: UniverseConfig,
    actions: pl.DataFrame | None = None,
) -> pl.DataFrame:
    frames = [
        membership_asof(bars, master, t, config, actions=actions).with_columns(
            pl.lit(t).alias("asof")
        )
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
