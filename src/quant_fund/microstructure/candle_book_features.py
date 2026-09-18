"""Fuse candlestick features with order-book metrics at bar timestamps."""

from __future__ import annotations

import polars as pl

from quant_fund.microstructure.book_panel import (
    validate_book_panel,
    validate_book_panel_depth_honesty,
)
from quant_fund.microstructure.synthetic_lob import synthesize_l2_from_bars
from quant_fund.northset.candles import candle_geometry


def candle_features_from_bars(bars: pl.DataFrame) -> pl.DataFrame:
    """Classic candlestick geometry features from OHLCV (Northset geometry)."""
    geo = candle_geometry(bars)
    volume = (
        pl.col("volume").alias("candle_volume")
        if "volume" in geo.columns
        else pl.lit(0.0).alias("candle_volume")
    )
    return geo.select(
        "security_id",
        "event_time",
        (
            pl.col("available_time") if "available_time" in geo.columns else pl.col("event_time")
        ).alias("decision_time"),
        pl.col("open").alias("candle_open"),
        pl.col("high").alias("candle_high"),
        pl.col("low").alias("candle_low"),
        pl.col("close").alias("candle_close"),
        (pl.col("return_close") if "return_close" in geo.columns else pl.col("close")).alias(
            "candle_return_close"
        ),
        volume,
        "candle_body_ret",
        "candle_body_frac",
        "candle_upper_wick_frac",
        "candle_lower_wick_frac",
        "wick_skew",
        "candle_range_frac",
        "candle_direction",
        "candle_gap",
        "close_location_value",
        "candle_doji",
        "candle_hammer",
        "candle_engulfing",
        "candle_spinning_top",
        "candle_marubozu",
        "candle_shooting_star",
    )


def attach_candle_book_features(
    bars: pl.DataFrame,
    *,
    book: pl.DataFrame | None = None,
    depth: int = 5,
    seed: int = 7,
    min_join_coverage: float | None = None,
    max_book_age_seconds: int = 86_400,
) -> pl.DataFrame:
    """As-of join candles to the latest book available by decision time.

    If ``book`` is None, synthesizes L2 from bars (research/SYNTHETIC path).
    Adds interaction features that only exist when both views are present.

    Honesty: always stamps ``book_source`` and ``book_dgp`` on the fused frame
    so downstream benches cannot silently mix SYNTHETIC L2 with vendor panels.
    Fail-closed on empty joins and on join coverage below ``min_join_coverage``
    (default 1.0 for SYNTHETIC L2, 0.5 for external panels).
    """
    candles = candle_features_from_bars(bars)
    synthesized = book is None
    book_df = book if book is not None else synthesize_l2_from_bars(bars, depth=depth, seed=seed)
    book_df = validate_book_panel(book_df)
    join_keys = ["security_id", "event_time"]
    for key in join_keys:
        if key not in book_df.columns:
            raise ValueError(f"book missing join key: {key}")
    if "available_time" not in book_df.columns:
        raise ValueError("book missing PIT column: available_time")
    if max_book_age_seconds < 1:
        raise ValueError("max_book_age_seconds must be positive")
    if synthesized:
        book_source = "synthetic_lob"
        book_dgp = "synthetic_lob"
    else:
        if "source" not in book_df.columns:
            raise ValueError(
                "external book panel missing required 'source' column "
                "(stamp vendor/synthetic before fuse — fail-closed honesty)"
            )
        sources = (
            book_df.select(pl.col("source").cast(pl.Utf8))
            .unique()
            .to_series()
            .drop_nulls()
            .to_list()
        )
        if not sources:
            raise ValueError("external book panel has empty/null source values")
        if len(sources) > 1:
            raise ValueError(
                f"external book panel mixes sources {sources!r} — refuse silent DGP mix"
            )
        book_source = str(sources[0])
        book_dgp = (
            "external_panel"
            if book_source.lower() not in {"synthetic", "synthetic_lob"}
            else "synthetic_lob"
        )
        validate_book_panel_depth_honesty(book_df)
    n_candle_rows = int(candles.height)
    book_slim = book_df.drop("revision_id", strict=False).rename(
        {
            "event_time": "book_event_time",
            "available_time": "book_available_time",
        }
    )
    fused = (
        candles.sort("decision_time")
        .join_asof(
            book_slim.sort("book_available_time"),
            left_on="decision_time",
            right_on="book_available_time",
            by="security_id",
            strategy="backward",
            tolerance=f"{int(max_book_age_seconds)}s",
            check_sortedness=False,
        )
        .filter(pl.col("book_available_time").is_not_null())
        .sort(["security_id", "event_time"])
    )
    if fused.height == 0:
        raise ValueError("candle/book join produced empty frame (timestamp mismatch?)")
    join_coverage = float(fused.height) / float(n_candle_rows) if n_candle_rows else 0.0
    if min_join_coverage is None:
        min_join_coverage = 1.0 if synthesized else 0.5
    if join_coverage + 1e-12 < float(min_join_coverage):
        raise ValueError(
            f"candle/book join coverage {join_coverage:.4f} < min_join_coverage "
            f"{float(min_join_coverage):.4f} (n_candles={n_candle_rows}, n_fused={fused.height})"
        )
    age_check = fused.select(
        (pl.col("decision_time") - pl.col("book_event_time"))
        .dt.total_seconds()
        .alias("book_age_seconds")
    )["book_age_seconds"]
    n_lookahead = int((age_check < -1e-9).sum())
    if n_lookahead:
        raise ValueError(
            f"PIT lookahead: book_event_time after decision_time on {n_lookahead} fused rows"
        )
    n_stale = int((age_check > float(max_book_age_seconds) + 1e-9).sum())
    if n_stale:
        raise ValueError(
            f"book age exceeds max_book_age_seconds={max_book_age_seconds} on {n_stale} fused rows"
        )
    # Top-of-book vendor panels often lack OFI/queue columns — derive research proxies.
    extras: list[pl.Expr] = []
    if "ofi" not in fused.columns and {"top_bid_size", "top_ask_size"} <= set(fused.columns):
        bid = pl.col("top_bid_size")
        ask = pl.col("top_ask_size")
        extras.append(
            (
                bid.cast(pl.Float64).diff().over("security_id").fill_null(0.0)
                - ask.cast(pl.Float64).diff().over("security_id").fill_null(0.0)
            ).alias("ofi")
        )
    if "queue_imbalance" not in fused.columns and "imbalance_top" in fused.columns:
        extras.append(pl.col("imbalance_top").alias("queue_imbalance"))
    if extras:
        fused = fused.with_columns(extras)
    # Normalize honesty columns (overwrite any joined source with canonical stamps).
    # Preserve book microprice_weight_balance (top-bid share) — never invent/overwrite.
    if "source" in fused.columns:
        fused = fused.drop("source")
    return fused.with_columns(
        pl.lit(book_source).alias("book_source"),
        pl.lit(book_dgp).alias("book_dgp"),
        pl.lit(join_coverage).alias("join_coverage"),
        (pl.col("decision_time") - pl.col("book_event_time"))
        .dt.total_seconds()
        .alias("book_age_seconds"),
        (pl.col("candle_direction") * pl.col("imbalance_top")).alias("candle_dir_x_imbalance"),
        (pl.col("candle_body_ret") - pl.col("microprice_minus_mid") / pl.col("candle_close")).alias(
            "candle_body_minus_micro_skew"
        ),
        (pl.col("spread_bps") * pl.col("candle_range_frac")).alias("spread_x_range"),
        # Candle close–mid diagnostic: 2·|C−mid|/mid (relative, ×2 convention).
        # Distinct from the book quoted/effective spread; fail-closed NaN if mid
        # is non-positive rather than minting an inf.
        (
            pl.when(pl.col("mid") > 0)
            .then(2.0 * (pl.col("candle_close") - pl.col("mid")).abs() / pl.col("mid"))
            .otherwise(None)
        ).alias("close_mid_abs_rel"),
        (pl.col("imbalance_depth") * pl.col("candle_body_frac")).alias("imbalance_x_body_frac"),
        (pl.col("candle_lower_wick_frac") * pl.col("spread_bps")).alias("lower_wick_x_spread"),
        (pl.col("imbalance_top") * pl.col("candle_range_frac")).alias("imbalance_x_range"),
        (
            pl.col("candle_direction")
            * pl.col("candle_volume").cast(pl.Float64)
            * pl.col("imbalance_top")
        ).alias("signed_vol_x_imbalance"),
    )
