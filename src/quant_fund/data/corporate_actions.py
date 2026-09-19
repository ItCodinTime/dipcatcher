"""Split, dividend, delist, and ticker-change adjustments. Raw prices are never overwritten."""

from __future__ import annotations

from datetime import datetime

import polars as pl

from quant_fund.schemas.errors import PointInTimeError

_DIVIDEND_ACTION_TYPES = ("cash_dividend", "special_dividend")
_TICKER_CHANGE = "ticker_change"
_DELIST = "delist"


def _available_action_pairs(bars: pl.DataFrame, actions: pl.DataFrame) -> pl.DataFrame:
    if "available_time" not in actions.columns:
        return actions.with_columns(pl.col("event_time").alias("bar_time"))
    return (
        bars.select(["security_id", pl.col("event_time").alias("bar_time")])
        .join(actions, on="security_id", how="inner")
        .filter(pl.col("available_time") <= pl.col("bar_time"))
    )


def _require_valid_split_factors(splits: pl.DataFrame) -> None:
    if splits.is_empty():
        return
    if "factor" not in splits.columns:
        raise PointInTimeError("split actions missing factor")
    factor = pl.col("factor").cast(pl.Float64, strict=False)
    bad = splits.filter(factor.is_null() | (~factor.is_finite()) | (factor <= 0))
    if bad.height:
        raise PointInTimeError("split actions contain non-positive or non-finite factors")


def _dividend_amounts(actions: pl.DataFrame) -> pl.DataFrame:
    if "amount" not in actions.columns:
        raise PointInTimeError("dividend actions missing amount")
    amount = pl.col("amount").cast(pl.Float64, strict=False)
    bad = actions.filter(amount.is_null() | (~amount.is_finite()) | (amount < 0))
    if bad.height:
        raise PointInTimeError("dividend actions contain negative or non-finite amounts")
    return (
        actions.select(
            "security_id",
            "event_time",
            amount.alias("dividend"),
        )
        .group_by(["security_id", "event_time"])
        .agg(pl.col("dividend").sum())
    )


def require_valid_ticker_changes(actions: pl.DataFrame) -> None:
    """Fail closed when ticker_change rows omit a non-blank ``new_ticker``."""
    if actions.is_empty() or "action_type" not in actions.columns:
        return
    changes = actions.filter(pl.col("action_type") == _TICKER_CHANGE)
    if changes.is_empty():
        return
    if "new_ticker" not in actions.columns:
        raise PointInTimeError("ticker_change actions missing new_ticker")
    ticker = pl.col("new_ticker").cast(pl.String, strict=False).str.strip_chars()
    bad = changes.filter(ticker.is_null() | (ticker == ""))
    if bad.height:
        raise PointInTimeError("ticker_change actions contain blank new_ticker")


def _visible_at_bar(actions: pl.DataFrame, event_time_col: str = "event_time") -> pl.Expr:
    known = pl.col("action_event") <= pl.col(event_time_col)
    if "available_time" not in actions.columns:
        return known
    return known & (pl.col("action_available") <= pl.col(event_time_col))


def ticker_overrides_asof(actions: pl.DataFrame, when: datetime) -> pl.DataFrame:
    """Latest PIT-visible ticker_change per security at ``when``."""
    require_valid_ticker_changes(actions)
    empty = pl.DataFrame(
        schema={"security_id": pl.String, "new_ticker": pl.String},
    )
    if actions.is_empty() or "action_type" not in actions.columns:
        return empty
    changes = actions.filter(pl.col("action_type") == _TICKER_CHANGE)
    if changes.is_empty():
        return empty
    visible = changes.filter(pl.col("event_time") <= when)
    if "available_time" in visible.columns:
        visible = visible.filter(pl.col("available_time") <= when)
    if visible.is_empty():
        return empty
    ticker = pl.col("new_ticker").cast(pl.String, strict=False).str.strip_chars()
    return visible.sort("event_time").group_by("security_id").agg(ticker.last().alias("new_ticker"))


def delisted_ids_asof(
    actions: pl.DataFrame,
    when: datetime,
    *,
    include_delisted: bool = True,
) -> list[str]:
    """Security ids whose delist is PIT-visible at ``when``.

    ``include_delisted=True`` (default) excludes a name only after the economic
    delist event is knowable. ``False`` excludes as soon as the announcement is
    available, even if the event is still in the future.
    """
    if actions.is_empty() or "action_type" not in actions.columns:
        return []
    delists = actions.filter(pl.col("action_type") == _DELIST)
    if delists.is_empty():
        return []
    if "available_time" in delists.columns:
        visible = delists.filter(pl.col("available_time") <= when)
    else:
        visible = delists.filter(pl.col("event_time") <= when)
    if include_delisted:
        # Last listed session remains eligible; names leave after the event.
        visible = visible.filter(pl.col("event_time") < when)
    if visible.is_empty():
        return []
    return visible.get_column("security_id").unique().to_list()


def apply_listing_actions(
    bars: pl.DataFrame,
    actions: pl.DataFrame,
    *,
    include_delisted: bool = True,
) -> pl.DataFrame:
    """Overlay PIT ticker_change and drop bars after a visible delist.

    Late-arriving listing events cannot rewrite bars whose ``event_time`` is
    earlier than ``available_time``. Direct callers missing ``action_type`` are
    a documented no-op, matching split/dividend adjustment.
    """
    if bars.is_empty() or actions.is_empty() or "action_type" not in actions.columns:
        return bars
    require_valid_ticker_changes(actions)
    out = _overlay_ticker_changes(bars, actions)
    return _drop_delisted_bars(out, actions, include_delisted=include_delisted)


def _overlay_ticker_changes(bars: pl.DataFrame, actions: pl.DataFrame) -> pl.DataFrame:
    changes = actions.filter(pl.col("action_type") == _TICKER_CHANGE)
    if changes.is_empty():
        return bars
    ticker = pl.col("new_ticker").cast(pl.String, strict=False).str.strip_chars()
    right_cols = [
        "security_id",
        pl.col("event_time").alias("action_event"),
        ticker.alias("action_ticker"),
    ]
    if "available_time" in changes.columns:
        right_cols.append(pl.col("available_time").alias("action_available"))
    right = changes.select(right_cols)
    keys = bars.select(["security_id", "event_time"])
    joined = keys.join(right, on="security_id", how="inner")
    picked = (
        joined.filter(_visible_at_bar(changes))
        .sort("action_event")
        .unique(subset=["security_id", "event_time"], keep="last", maintain_order=True)
        .select(["security_id", "event_time", "action_ticker"])
    )
    out = bars.join(picked, on=["security_id", "event_time"], how="left")
    if "symbol" in out.columns:
        out = out.with_columns(
            pl.coalesce(pl.col("action_ticker"), pl.col("symbol")).alias("symbol")
        )
    else:
        out = out.rename({"action_ticker": "symbol"})
        return out
    return out.drop("action_ticker")


def _drop_delisted_bars(
    bars: pl.DataFrame,
    actions: pl.DataFrame,
    *,
    include_delisted: bool,
) -> pl.DataFrame:
    delists = actions.filter(pl.col("action_type") == _DELIST)
    if delists.is_empty():
        return bars
    right_cols = ["security_id", pl.col("event_time").alias("action_event")]
    if "available_time" in delists.columns:
        right_cols.append(pl.col("available_time").alias("action_available"))
    joined = bars.select(["security_id", "event_time"]).join(
        delists.select(right_cols),
        on="security_id",
        how="inner",
    )
    announced = (
        pl.col("action_available") <= pl.col("event_time")
        if "available_time" in delists.columns
        else pl.col("action_event") <= pl.col("event_time")
    )
    drop_mask = (
        announced & (pl.col("event_time") > pl.col("action_event"))
        if include_delisted
        else announced
    )
    drop = joined.filter(drop_mask).select(["security_id", "event_time"]).unique()
    if drop.is_empty():
        return bars
    return bars.join(drop, on=["security_id", "event_time"], how="anti")


def cumulative_split_factors(bars: pl.DataFrame, actions: pl.DataFrame) -> pl.DataFrame:
    """Backward-looking split factor so the last bar has factor 1.0.

    A 2-for-1 split on ex-date D means pre-split prices are multiplied by 1/2
    when constructing a split-adjusted series that matches post-split quotes.
    """
    if actions.is_empty() or "action_type" not in actions.columns:
        return bars.with_columns(pl.lit(1.0).alias("split_factor"))

    if "available_time" not in actions.columns:
        splits = actions.filter(pl.col("action_type") == "split")
        if splits.is_empty():
            return bars.with_columns(pl.lit(1.0).alias("split_factor"))
        _require_valid_split_factors(splits)
        splits = splits.select(
            "security_id",
            pl.col("event_time").alias("ex_time"),
            pl.col("factor").alias("split_size"),
        )
        joined = bars.join(splits, on="security_id", how="left")
        joined = joined.with_columns(
            pl.when(pl.col("ex_time").is_not_null() & (pl.col("event_time") < pl.col("ex_time")))
            .then(pl.col("split_size").fill_null(1.0))
            .otherwise(1.0)
            .alias("_one_split")
        )
        factors = joined.group_by(["security_id", "event_time"]).agg(
            pl.col("_one_split").product().alias("split_factor")
        )
    else:
        pairs = _available_action_pairs(bars, actions).filter(
            (pl.col("action_type") == "split") & (pl.col("bar_time") < pl.col("event_time"))
        )
        if pairs.is_empty():
            return bars.with_columns(pl.lit(1.0).alias("split_factor"))
        _require_valid_split_factors(pairs)
        factors = (
            pairs.group_by(["security_id", "bar_time"])
            .agg(pl.col("factor").product().alias("split_factor"))
            .rename({"bar_time": "event_time"})
        )
    return bars.join(factors, on=["security_id", "event_time"], how="left").with_columns(
        pl.col("split_factor").fill_null(1.0)
    )


def attach_dividends(bars: pl.DataFrame, actions: pl.DataFrame) -> pl.DataFrame:
    if actions.is_empty() or "action_type" not in actions.columns:
        return bars.with_columns(pl.lit(0.0).alias("dividend"))
    if "available_time" not in actions.columns:
        raw = actions.filter(pl.col("action_type").is_in(list(_DIVIDEND_ACTION_TYPES)))
    else:
        raw = _available_action_pairs(bars, actions).filter(
            pl.col("action_type").is_in(list(_DIVIDEND_ACTION_TYPES))
            & (pl.col("bar_time") == pl.col("event_time"))
        )
    if raw.is_empty():
        return bars.with_columns(pl.lit(0.0).alias("dividend"))
    divs = _dividend_amounts(raw)
    return bars.join(divs, on=["security_id", "event_time"], how="left").with_columns(
        pl.col("dividend").fill_null(0.0)
    )


def adjust_prices(bars: pl.DataFrame, actions: pl.DataFrame) -> pl.DataFrame:
    """Add split-adjusted OHLC and a total-return close.

    Total return close reinvests cash dividends at the ex-date close.
    """
    out = cumulative_split_factors(bars, actions)
    out = attach_dividends(out, actions)
    sf = pl.col("split_factor")
    out = out.with_columns(
        (pl.col("open") / sf).alias("open_split_adjusted"),
        (pl.col("high") / sf).alias("high_split_adjusted"),
        (pl.col("low") / sf).alias("low_split_adjusted"),
        (pl.col("close") / sf).alias("close_split_adjusted"),
        (pl.col("volume") * sf).alias("volume_split_adjusted"),
    )
    # TR index: start at first split-adjusted close; multiply by (1 + div/close)
    out = out.sort(["security_id", "event_time"]).with_columns(
        (pl.col("dividend") / pl.col("close")).fill_nan(0.0).alias("_div_ret")
    )
    out = out.with_columns(
        (
            pl.col("close_split_adjusted")
            * (1.0 + pl.col("_div_ret")).cum_prod().over("security_id")
            / (1.0 + pl.col("_div_ret"))
        ).alias("close_total_return")
    )
    # For bars with no dividend, TR close equals split-adjusted close after scaling
    # Use a cleaner definition: TR return = simple return of split-adj close plus div/prev_close
    prev = pl.col("close_split_adjusted").shift(1).over("security_id")
    prev_raw = pl.col("close").shift(1).over("security_id")
    simple = pl.col("close_split_adjusted") / prev - 1.0
    div_simple = pl.col("dividend") / prev_raw
    tr = simple + div_simple.fill_null(0.0)
    out = out.with_columns(tr.fill_null(0.0).alias("_tr_ret"))
    out = out.with_columns(
        (
            pl.col("close_split_adjusted").first().over("security_id")
            * (1.0 + pl.col("_tr_ret")).cum_prod().over("security_id")
        ).alias("close_total_return")
    )
    return out.drop(["_div_ret", "_tr_ret"])
