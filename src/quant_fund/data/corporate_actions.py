"""Split and dividend adjustments. Raw prices are never overwritten."""

from __future__ import annotations

import polars as pl


def cumulative_split_factors(bars: pl.DataFrame, actions: pl.DataFrame) -> pl.DataFrame:
    """Backward-looking split factor so the last bar has factor 1.0.

    A 2-for-1 split on ex-date D means pre-split prices are multiplied by 1/2
    when constructing a split-adjusted series that matches post-split quotes.
    """
    if actions.is_empty() or "action_type" not in actions.columns:
        return bars.with_columns(pl.lit(1.0).alias("split_factor"))

    splits = actions.filter(pl.col("action_type") == "split").select(
        "security_id",
        pl.col("event_time").alias("ex_time"),
        pl.col("factor").alias("split_size"),
    )
    if splits.is_empty():
        return bars.with_columns(pl.lit(1.0).alias("split_factor"))

    joined = bars.join(splits, on="security_id", how="left")
    # factor applies to bars strictly before ex_time
    joined = joined.with_columns(
        pl.when(pl.col("ex_time").is_not_null() & (pl.col("event_time") < pl.col("ex_time")))
        .then(pl.col("split_size").fill_null(1.0))
        .otherwise(1.0)
        .alias("_one_split")
    )
    factors = joined.group_by(["security_id", "event_time"]).agg(
        pl.col("_one_split").product().alias("split_factor")
    )
    return bars.join(factors, on=["security_id", "event_time"], how="left").with_columns(
        pl.col("split_factor").fill_null(1.0)
    )


def attach_dividends(bars: pl.DataFrame, actions: pl.DataFrame) -> pl.DataFrame:
    if actions.is_empty():
        return bars.with_columns(pl.lit(0.0).alias("dividend"))
    divs = actions.filter(
        pl.col("action_type").is_in(["cash_dividend", "special_dividend"])
    ).select(
        "security_id",
        pl.col("event_time"),
        pl.col("amount").fill_null(0.0).alias("dividend"),
    )
    if divs.is_empty():
        return bars.with_columns(pl.lit(0.0).alias("dividend"))
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
