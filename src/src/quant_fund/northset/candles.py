"""Candlestick geometry flags for Northset. Not trading signals."""

from __future__ import annotations

from typing import cast

import polars as pl

_DOJI_BODY = 0.10
_HAMMER_LOWER = 0.60
_HAMMER_BODY = 0.30
_HAMMER_UPPER = 0.15
_SPIN_BODY = 0.30
_SPIN_WICK = 0.20
_MARUBOZU_BODY = 0.90
_STAR_UPPER = 0.60
_STAR_BODY = 0.30
_STAR_LOWER = 0.15


def candle_geometry(bars: pl.DataFrame) -> pl.DataFrame:
    """OHLC geometry plus gap, doji, hammer, and engulfing flags.

    Flags are shape descriptors for research IC, not pattern-trading claims.
    """
    required = ("security_id", "event_time", "open", "high", "low", "close")
    missing = [c for c in required if c not in bars.columns]
    if missing:
        raise ValueError(f"bars missing required columns: {missing}")
    if bars.height == 0:
        raise ValueError("bars must be non-empty")
    rng = (pl.col("high") - pl.col("low")).clip(lower_bound=1e-12)
    body = pl.col("close") - pl.col("open")
    body_frac = body.abs() / rng
    upper = (pl.col("high") - pl.max_horizontal(pl.col("open"), pl.col("close"))) / rng
    lower = (pl.min_horizontal(pl.col("open"), pl.col("close")) - pl.col("low")) / rng
    df = bars.sort(["security_id", "event_time"]).with_columns(
        pl.col("close").shift(1).over("security_id").alias("prev_close"),
        pl.col("open").shift(1).over("security_id").alias("prev_open"),
    )
    prev_body = pl.col("prev_close") - pl.col("prev_open")
    bull_engulf = (
        (body > 0.0)
        & (prev_body < 0.0)
        & (pl.col("open") <= pl.col("prev_close"))
        & (pl.col("close") >= pl.col("prev_open"))
    )
    bear_engulf = (
        (body < 0.0)
        & (prev_body > 0.0)
        & (pl.col("open") >= pl.col("prev_close"))
        & (pl.col("close") <= pl.col("prev_open"))
    )
    return df.with_columns(
        (body / pl.col("close")).alias("candle_body_ret"),
        body_frac.alias("candle_body_frac"),
        upper.alias("candle_upper_wick_frac"),
        lower.alias("candle_lower_wick_frac"),
        (lower - upper).alias("wick_skew"),
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("candle_range_frac"),
        pl.when(pl.col("close") > pl.col("open"))
        .then(pl.lit(1.0))
        .when(pl.col("close") < pl.col("open"))
        .then(pl.lit(-1.0))
        .otherwise(pl.lit(0.0))
        .alias("candle_direction"),
        (pl.col("open") / pl.col("prev_close") - 1.0).alias("candle_gap"),
        ((2.0 * pl.col("close") - pl.col("high") - pl.col("low")) / rng).alias(
            "close_location_value"
        ),
        (body_frac < _DOJI_BODY).cast(pl.Float64).alias("candle_doji"),
        ((lower > _HAMMER_LOWER) & (body_frac < _HAMMER_BODY) & (upper < _HAMMER_UPPER))
        .cast(pl.Float64)
        .alias("candle_hammer"),
        (bull_engulf | bear_engulf).cast(pl.Float64).alias("candle_engulfing"),
        ((body_frac < _SPIN_BODY) & (upper > _SPIN_WICK) & (lower > _SPIN_WICK))
        .cast(pl.Float64)
        .alias("candle_spinning_top"),
        (body_frac > _MARUBOZU_BODY).cast(pl.Float64).alias("candle_marubozu"),
        ((upper > _STAR_UPPER) & (body_frac < _STAR_BODY) & (lower < _STAR_LOWER))
        .cast(pl.Float64)
        .alias("candle_shooting_star"),
    )


_FLAG_RATES = (
    ("candle_doji", "doji_rate"),
    ("candle_hammer", "hammer_rate"),
    ("candle_engulfing", "engulfing_rate"),
    ("candle_spinning_top", "spinning_top_rate"),
    ("candle_marubozu", "marubozu_rate"),
    ("candle_shooting_star", "shooting_star_rate"),
)


def geometry_rates(frame: pl.DataFrame) -> dict[str, float]:
    """Mean rates of geometry flags. Empty / missing → NaN."""
    out: dict[str, float] = {}
    for col, key in _FLAG_RATES:
        if col not in frame.columns or frame.height == 0:
            out[key] = float("nan")
            continue
        mean = frame[col].mean()
        out[key] = float(cast(float, mean)) if mean is not None else float("nan")
    return out
