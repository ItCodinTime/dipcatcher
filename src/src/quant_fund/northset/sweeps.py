"""Liquidity-sweep research: prior-extreme takeouts, reclaim vs follow-through.

A sweep is a shape event on PIT daily bars: price trades beyond the prior
rolling extreme (stops/liquidity above the old high or below the old low).
What matters scientifically is the *resolution* — reclaim (close back inside
the prior range) vs follow-through (close beyond). These are descriptors for
date-level IC, not trade signals. No Sharpe. SYNTHETIC labeled upstream.
"""

from __future__ import annotations

from typing import cast

import polars as pl

from quant_fund.northset.identities import ohlc_identity_frame

_SWEEP_OUT_COLS = (
    "sweep_high",
    "sweep_low",
    "sweep_both",
    "sweep_high_reclaim",
    "sweep_low_reclaim",
    "sweep_high_follow",
    "sweep_low_follow",
    "sweep_depth_high",
    "sweep_depth_low",
    "sweep_reject_signed",
    "sweep_follow_signed",
    "sweep_depth_signed",
)


def sweep_output_columns() -> tuple[str, ...]:
    """Columns produced by :func:`liquidity_sweep_frame` (join-safe subset)."""
    return _SWEEP_OUT_COLS


def liquidity_sweep_frame(bars: pl.DataFrame, *, lookback: int = 20) -> pl.DataFrame:
    """Flag sweeps of the prior ``lookback``-bar extreme and their resolution.

    PIT: the reference extreme uses only bars strictly before the current one
    (shifted rolling max/min). Rows without a full lookback window carry null
    flags; ``sweep_eligible`` marks rows with a defined prior extreme.
    """
    required = ("security_id", "event_time", "open", "high", "low", "close")
    missing = [c for c in required if c not in bars.columns]
    if missing:
        raise ValueError(f"bars missing required columns: {missing}")
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    if bars.height == 0:
        raise ValueError("bars must be non-empty")
    frame = ohlc_identity_frame(bars.sort(["security_id", "event_time"]))
    if "volume" in frame.columns:
        frame = frame.with_columns(
            (pl.col("ohlc_ok") & pl.col("volume").is_finite() & (pl.col("volume") >= 0.0)).alias(
                "ohlc_ok"
            )
        )
    # Invalid prints must not enter the PIT rolling extreme (a bogus high would
    # become prior_high and mint false later sweeps). Extremes are rolling max/min
    # over the last ``lookback`` *valid* bars, not calendar slots that include holes.
    valid_ext = (
        frame.filter(pl.col("ohlc_ok"))
        .select("security_id", "event_time", "high", "low")
        .with_columns(
            pl.col("high")
            .shift(1)
            .rolling_max(window_size=int(lookback), min_samples=int(lookback))
            .over("security_id")
            .alias("prior_high"),
            pl.col("low")
            .shift(1)
            .rolling_min(window_size=int(lookback), min_samples=int(lookback))
            .over("security_id")
            .alias("prior_low"),
        )
        .select("security_id", "event_time", "prior_high", "prior_low")
    )
    frame = frame.join(valid_ext, on=["security_id", "event_time"], how="left")
    hi = pl.when(pl.col("ohlc_ok")).then(pl.col("high")).otherwise(None)
    lo = pl.when(pl.col("ohlc_ok")).then(pl.col("low")).otherwise(None)
    cl = pl.when(pl.col("ohlc_ok")).then(pl.col("close")).otherwise(None)
    frame = frame.with_columns(
        (pl.col("prior_high").is_not_null() & pl.col("ohlc_ok")).alias("sweep_eligible"),
        (hi > pl.col("prior_high")).cast(pl.Float64).alias("sweep_high"),
        (lo < pl.col("prior_low")).cast(pl.Float64).alias("sweep_low"),
    )
    frame = frame.with_columns(
        ((pl.col("sweep_high") == 1.0) & (pl.col("sweep_low") == 1.0))
        .cast(pl.Float64)
        .alias("sweep_both")
    )
    unambiguous = pl.col("sweep_both").fill_null(0.0) == 0.0
    frame = frame.with_columns(
        (unambiguous & (pl.col("sweep_high") == 1.0) & (cl < pl.col("prior_high")))
        .cast(pl.Float64)
        .alias("sweep_high_reclaim"),
        (unambiguous & (pl.col("sweep_low") == 1.0) & (cl > pl.col("prior_low")))
        .cast(pl.Float64)
        .alias("sweep_low_reclaim"),
        (unambiguous & (pl.col("sweep_high") == 1.0) & (cl >= pl.col("prior_high")))
        .cast(pl.Float64)
        .alias("sweep_high_follow"),
        (unambiguous & (pl.col("sweep_low") == 1.0) & (cl <= pl.col("prior_low")))
        .cast(pl.Float64)
        .alias("sweep_low_follow"),
    )
    frame = frame.with_columns(
        pl.when((pl.col("sweep_high").fill_null(0.0) == 1.0) & pl.col("ohlc_ok"))
        .then((pl.col("high") - pl.col("prior_high")) / pl.col("prior_high"))
        .otherwise(0.0)
        .alias("sweep_depth_high"),
        pl.when((pl.col("sweep_low").fill_null(0.0) == 1.0) & pl.col("ohlc_ok"))
        .then((pl.col("prior_low") - pl.col("low")) / pl.col("prior_low"))
        .otherwise(0.0)
        .alias("sweep_depth_low"),
    )
    return frame.with_columns(
        pl.when(pl.col("sweep_low_reclaim").fill_null(0.0) == 1.0)
        .then(pl.col("sweep_depth_low"))
        .when(pl.col("sweep_high_reclaim").fill_null(0.0) == 1.0)
        .then(-pl.col("sweep_depth_high"))
        .otherwise(0.0)
        .alias("sweep_reject_signed"),
        pl.when(pl.col("sweep_high_follow").fill_null(0.0) == 1.0)
        .then(pl.col("sweep_depth_high"))
        .when(pl.col("sweep_low_follow").fill_null(0.0) == 1.0)
        .then(-pl.col("sweep_depth_low"))
        .otherwise(0.0)
        .alias("sweep_follow_signed"),
        pl.when(pl.col("sweep_both").fill_null(0.0) == 1.0)
        .then(0.0)
        .when(pl.col("sweep_high").fill_null(0.0) == 1.0)
        .then(pl.col("sweep_depth_high"))
        .when(pl.col("sweep_low").fill_null(0.0) == 1.0)
        .then(-pl.col("sweep_depth_low"))
        .otherwise(0.0)
        .alias("sweep_depth_signed"),
    )


def _rate(frame: pl.DataFrame, num: pl.Expr, denom_mask: pl.Expr) -> float:
    sub = frame.filter(denom_mask)
    if sub.height == 0:
        return float("nan")
    value = sub.select(num.alias("_r"))["_r"].mean()
    return float(cast(float, value)) if value is not None else float("nan")


def sweep_rates(frame: pl.DataFrame) -> dict[str, float]:
    """Sweep event rates. Eligible = full lookback history (PIT)."""
    if frame.height == 0 or "sweep_eligible" not in frame.columns:
        return {key: float("nan") for key in _RATE_KEYS}
    eligible = pl.col("sweep_eligible")
    out: dict[str, float] = {
        "sweep_eligible_rate": _rate(frame, eligible.cast(pl.Float64).mean(), pl.lit(True)),
        "sweep_high_rate": _rate(frame, pl.col("sweep_high").mean(), eligible),
        "sweep_low_rate": _rate(frame, pl.col("sweep_low").mean(), eligible),
        "sweep_both_rate": _rate(frame, pl.col("sweep_both").mean(), eligible),
        "sweep_any_rate": _rate(
            frame,
            pl.max_horizontal(pl.col("sweep_high"), pl.col("sweep_low")).mean(),
            eligible,
        ),
        "sweep_high_reclaim_share": _rate(
            frame,
            pl.col("sweep_high_reclaim").mean(),
            (pl.col("sweep_high") == 1.0) & (pl.col("sweep_both") == 0.0),
        ),
        "sweep_low_reclaim_share": _rate(
            frame,
            pl.col("sweep_low_reclaim").mean(),
            (pl.col("sweep_low") == 1.0) & (pl.col("sweep_both") == 0.0),
        ),
        "sweep_high_follow_share": _rate(
            frame,
            pl.col("sweep_high_follow").mean(),
            (pl.col("sweep_high") == 1.0) & (pl.col("sweep_both") == 0.0),
        ),
        "sweep_low_follow_share": _rate(
            frame,
            pl.col("sweep_low_follow").mean(),
            (pl.col("sweep_low") == 1.0) & (pl.col("sweep_both") == 0.0),
        ),
        "mean_sweep_depth_high": _rate(
            frame, pl.col("sweep_depth_high").mean(), pl.col("sweep_high") == 1.0
        ),
        "mean_sweep_depth_low": _rate(
            frame, pl.col("sweep_depth_low").mean(), pl.col("sweep_low") == 1.0
        ),
    }
    return out


_RATE_KEYS = (
    "sweep_eligible_rate",
    "sweep_high_rate",
    "sweep_low_rate",
    "sweep_both_rate",
    "sweep_any_rate",
    "sweep_high_reclaim_share",
    "sweep_low_reclaim_share",
    "sweep_high_follow_share",
    "sweep_low_follow_share",
    "mean_sweep_depth_high",
    "mean_sweep_depth_low",
)
