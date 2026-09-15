"""Return, momentum, reversal, volatility, liquidity, and market features."""

from __future__ import annotations

from datetime import datetime

import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.data.point_in_time import validate_feature_frame
from quant_fund.features.cross_sectional import apply_cross_sectional
from quant_fund.features.metadata import FEATURE_SET_VERSION, FeatureMetadata

PX = "close_total_return"
RAW_PX = "close"


def _log_ret(horizon: int) -> pl.Expr:
    prev = pl.col(PX).shift(horizon).over("security_id")
    return (pl.col(PX) / prev).log().alias(f"log_ret_{horizon}")


def _simple_ret(horizon: int) -> pl.Expr:
    prev = pl.col(PX).shift(horizon).over("security_id")
    return (pl.col(PX) / prev - 1.0).alias(f"ret_{horizon}")


def compute_base_features(bars: pl.DataFrame, config: AppConfig) -> pl.DataFrame:
    lam = config.features.ewma_lambda
    df = bars.sort(["security_id", "event_time"])
    df = df.with_columns(
        _simple_ret(1),
        _simple_ret(5),
        _simple_ret(20),
        _log_ret(1),
        _log_ret(5),
        _log_ret(20),
        (
            pl.col("open_split_adjusted")
            / pl.col("close_split_adjusted").shift(1).over("security_id")
            - 1.0
        ).alias("ret_overnight"),
        (pl.col("close_split_adjusted") / pl.col("open_split_adjusted") - 1.0).alias(
            "ret_open_close"
        ),
    )
    # momentum
    for h in (5, 20, 60, 126, 252):
        df = df.with_columns(_simple_ret(h).alias(f"mom_{h}"))
    # 12-1: 252d minus last 21d
    df = df.with_columns(
        (
            (pl.col(PX) / pl.col(PX).shift(252).over("security_id") - 1.0)
            - (pl.col(PX) / pl.col(PX).shift(21).over("security_id") - 1.0)
        ).alias("mom_12_1"),
        (-pl.col("ret_1")).alias("reversal_1"),
        (
            (pl.col(PX) - pl.col(PX).rolling_mean(20).over("security_id"))
            / pl.col(PX).rolling_std(20).over("security_id")
        ).alias("z_vs_ma20"),
    )
    # vol
    r = pl.col("log_ret_1")
    df = df.with_columns(
        r.rolling_std(20).over("security_id").alias("vol_20"),
        r.rolling_std(60).over("security_id").alias("vol_60"),
        r.ewm_std(alpha=1.0 - lam, adjust=False).over("security_id").alias("vol_ewma"),
    )
    # Parkinson / Garman-Klass use raw OHLC (split-adjusted levels)
    log_hl = (pl.col("high_split_adjusted") / pl.col("low_split_adjusted")).log()
    log_co = (pl.col("close_split_adjusted") / pl.col("open_split_adjusted")).log()
    park = (log_hl.pow(2) / (4.0 * pl.lit(2.0).log())).rolling_mean(20).over("security_id")
    gk = (
        (0.5 * log_hl.pow(2) - (2.0 * pl.lit(2.0).log() - 1.0) * log_co.pow(2))
        .rolling_mean(20)
        .over("security_id")
    )
    df = df.with_columns(
        park.sqrt().alias("vol_parkinson"),
        gk.clip(0.0).sqrt().alias("vol_garman_klass"),
        r.pow(2).rolling_mean(20).over("security_id").sqrt().alias("vol_realized"),
        r.rolling_std(20)
        .over("security_id")
        .rolling_std(20)
        .over("security_id")
        .alias("vol_of_vol"),
    )
    # liquidity
    dollar = pl.col(RAW_PX) * pl.col("volume")
    df = df.with_columns(
        dollar.alias("dollar_volume"),
        dollar.rolling_mean(config.features.adv_lookback).over("security_id").alias("adv"),
        (dollar / dollar.rolling_mean(config.features.adv_lookback).over("security_id")).alias(
            "rel_volume"
        ),
        (pl.col("ret_1").abs() / dollar)
        .rolling_mean(config.features.amihud_lookback)
        .over("security_id")
        .alias("amihud"),
        (pl.col("volume") / pl.col("volume").rolling_mean(20).over("security_id")).alias(
            "turnover_proxy"
        ),
        dollar.rolling_std(20).over("security_id").alias("volume_vol"),
    )
    return df


def add_market_features(df: pl.DataFrame, benchmark_id: str) -> pl.DataFrame:
    mkt = df.filter(pl.col("security_id") == benchmark_id).select(
        "event_time",
        pl.col("ret_1").alias("mkt_ret_1"),
        pl.col("vol_20").alias("mkt_vol_20"),
    )
    out = df.join(mkt, on="event_time", how="left")
    # cross-sectional dispersion / breadth at t (uses only today's cross-section)
    cs = out.group_by("event_time").agg(
        pl.col("ret_1").std().alias("cs_dispersion"),
        (pl.col("ret_1") > 0).mean().alias("breadth"),
        pl.col("ret_1").mean().alias("cs_mean_ret"),
    )
    out = out.join(cs, on="event_time", how="left")
    if "sector" in out.columns:
        sec = out.group_by(["event_time", "sector"]).agg(
            pl.col("ret_1").mean().alias("sector_ret_1")
        )
        out = out.join(sec, on=["event_time", "sector"], how="left")
        out = out.with_columns(
            (pl.col("mom_20") - pl.col("mom_20").mean().over(["event_time", "sector"])).alias(
                "sector_relative_mom_20"
            )
        )
    return out


def feature_catalog() -> list[FeatureMetadata]:
    cols = [
        ("ret_1", 1, "returns"),
        ("ret_5", 5, "returns"),
        ("ret_20", 20, "returns"),
        ("log_ret_1", 1, "returns"),
        ("mom_20", 20, "momentum"),
        ("mom_60", 60, "momentum"),
        ("mom_12_1", 252, "momentum"),
        ("reversal_1", 1, "reversal"),
        ("vol_20", 20, "volatility"),
        ("vol_ewma", 20, "volatility"),
        ("vol_parkinson", 20, "volatility"),
        ("amihud", 20, "liquidity"),
        ("adv", 20, "liquidity"),
        ("cs_z_ret_1", 1, "cross_sectional"),
        ("planted_signal", 0, "synthetic_oracle"),
        ("cs_z_planted_signal", 0, "synthetic_oracle"),
    ]
    return [
        FeatureMetadata(
            name=n,
            version=FEATURE_SET_VERSION,
            lookback=lb,
            required_frequency="1d",
            source_columns=["close_total_return", "volume"],
            family=fam,
        )
        for n, lb, fam in cols
    ]


def build_features(
    bars: pl.DataFrame,
    config: AppConfig,
    *,
    decision_time: datetime | None = None,
) -> pl.DataFrame:
    df = compute_base_features(bars, config)
    df = add_market_features(df, config.data.benchmark_id)
    names = [
        c
        for c in [
            "ret_1",
            "ret_5",
            "ret_20",
            "mom_5",
            "mom_20",
            "mom_60",
            "mom_12_1",
            "reversal_1",
            "z_vs_ma20",
            "vol_20",
            "vol_ewma",
            "vol_parkinson",
            "amihud",
            "adv",
            "rel_volume",
            "planted_signal",
        ]
        if c in df.columns
    ]
    sector = "sector" if "sector" in df.columns else None
    df = apply_cross_sectional(df, names, config.features.winsor_p, sector=sector)
    df = df.with_columns(
        pl.col("available_time").max().over("event_time").alias("max_source_available_time"),
        pl.col("event_time").alias("decision_time"),
        pl.lit(FEATURE_SET_VERSION).alias("feature_set_version"),
    )
    if decision_time is not None:
        validate_feature_frame(df.filter(pl.col("event_time") == decision_time), decision_time)
    return df
