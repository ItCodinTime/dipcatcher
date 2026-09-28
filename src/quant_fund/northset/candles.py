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
    # A bar's geometry is only defined when OHLC are finite, positive, and
    # consistent (high >= low, open/close inside the range). Flat bars
    # (high == low) have zero denominator — their geometry is undefined,
    # not ``x / 1e-12``. Clipping the range upward would flag corrupt or
    # degenerate bars (e.g. a nonzero body on a flat bar becomes
    # "marubozu"), so all range-derived columns are null on invalid bars.
    ohlc_ok = (
        pl.col("open").is_finite()
        & pl.col("high").is_finite()
        & pl.col("low").is_finite()
        & pl.col("close").is_finite()
        & (pl.col("open") > 0.0)
        & (pl.col("high") > 0.0)
        & (pl.col("low") > 0.0)
        & (pl.col("close") > 0.0)
        & (pl.col("high") >= pl.col("low"))
        & (pl.col("high") - pl.col("low") > 0.0)
    )
    rng = pl.col("high") - pl.col("low")
    body = pl.col("close") - pl.col("open")
    body_frac = body.abs() / rng
    upper = (pl.col("high") - pl.max_horizontal(pl.col("open"), pl.col("close"))) / rng
    lower = (pl.min_horizontal(pl.col("open"), pl.col("close")) - pl.col("low")) / rng
    df = bars.sort(["security_id", "event_time"]).with_columns(
        pl.col("close").shift(1).over("security_id").alias("prev_close"),
        pl.col("open").shift(1).over("security_id").alias("prev_open"),
    )
    prev_body = pl.col("prev_close") - pl.col("prev_open")
    prev_ok = pl.col("prev_open").is_finite() & pl.col("prev_close").is_finite()
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

    def _flag(expr: pl.Expr, needs_prev: bool = False) -> pl.Expr:
        ok = ohlc_ok & prev_ok if needs_prev else ohlc_ok
        return pl.when(ok).then(expr.cast(pl.Float64)).otherwise(None)

    return df.with_columns(
        pl.when(ohlc_ok).then(body / pl.col("close")).otherwise(None).alias("candle_body_ret"),
        pl.when(ohlc_ok).then(body_frac).otherwise(None).alias("candle_body_frac"),
        pl.when(ohlc_ok).then(upper).otherwise(None).alias("candle_upper_wick_frac"),
        pl.when(ohlc_ok).then(lower).otherwise(None).alias("candle_lower_wick_frac"),
        pl.when(ohlc_ok).then(lower - upper).otherwise(None).alias("wick_skew"),
        pl.when(ohlc_ok)
        .then((pl.col("high") - pl.col("low")) / pl.col("close"))
        .otherwise(None)
        .alias("candle_range_frac"),
        pl.when(ohlc_ok & (pl.col("close") > pl.col("open")))
        .then(pl.lit(1.0))
        .when(ohlc_ok & (pl.col("close") < pl.col("open")))
        .then(pl.lit(-1.0))
        .when(ohlc_ok)
        .then(pl.lit(0.0))
        .otherwise(None)
        .alias("candle_direction"),
        pl.when(ohlc_ok & pl.col("prev_close").is_finite() & (pl.col("prev_close") > 0.0))
        .then(pl.col("open") / pl.col("prev_close") - 1.0)
        .otherwise(None)
        .alias("candle_gap"),
        pl.when(ohlc_ok)
        .then((2.0 * pl.col("close") - pl.col("high") - pl.col("low")) / rng)
        .otherwise(None)
        .alias("close_location_value"),
        _flag(body_frac < _DOJI_BODY).alias("candle_doji"),
        _flag((lower > _HAMMER_LOWER) & (body_frac < _HAMMER_BODY) & (upper < _HAMMER_UPPER)).alias(
            "candle_hammer"
        ),
        _flag(bull_engulf | bear_engulf, needs_prev=True).alias("candle_engulfing"),
        _flag((body_frac < _SPIN_BODY) & (upper > _SPIN_WICK) & (lower > _SPIN_WICK)).alias(
            "candle_spinning_top"
        ),
        _flag(body_frac > _MARUBOZU_BODY).alias("candle_marubozu"),
        _flag((upper > _STAR_UPPER) & (body_frac < _STAR_BODY) & (lower < _STAR_LOWER)).alias(
            "candle_shooting_star"
        ),
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
