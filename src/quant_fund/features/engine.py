"""Return, momentum, reversal, volatility, liquidity, and market features."""

from __future__ import annotations

from datetime import datetime
from typing import cast

import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.data.point_in_time import validate_feature_frame
from quant_fund.data.universe import attach_membership_flag
from quant_fund.features.cross_sectional import apply_cross_sectional, decision_eligible_expr
from quant_fund.features.metadata import FEATURE_SET_VERSION, FeatureMetadata
from quant_fund.schemas.errors import PointInTimeError

PX = "close_total_return"
RAW_PX = "close"


def _log_ret(horizon: int) -> pl.Expr:
    prev = pl.col(PX).shift(horizon).over("security_id")
    return (pl.col(PX) / prev).log().alias(f"log_ret_{horizon}")


def _simple_ret(horizon: int) -> pl.Expr:
    prev = pl.col(PX).shift(horizon).over("security_id")
    return (pl.col(PX) / prev - 1.0).alias(f"ret_{horizon}")


_BASE_FEATURE_REQUIRED = (
    "security_id",
    "event_time",
    "close_total_return",
    "close",
    "volume",
    "open_split_adjusted",
    "close_split_adjusted",
    "high_split_adjusted",
    "low_split_adjusted",
)


def compute_base_features(bars: pl.DataFrame, config: AppConfig) -> pl.DataFrame:
    if bars.height == 0:
        raise ValueError("bars must be non-empty for compute_base_features")
    missing = [c for c in _BASE_FEATURE_REQUIRED if c not in bars.columns]
    if missing:
        raise ValueError(f"bars missing required OHLCV columns: {missing}")
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
    # 12-1: return from t-252 to t-21 (skip the most recent month).
    # Subtracting two simple returns would introduce a cross-term bias.
    df = df.with_columns(
        (
            pl.col(PX).shift(21).over("security_id") / pl.col(PX).shift(252).over("security_id")
            - 1.0
        ).alias("mom_12_1"),
        (-pl.col("ret_1")).alias("reversal_1"),
        (
            (pl.col(PX) - pl.col(PX).rolling_mean(20).over("security_id"))
            / pl.col(PX).rolling_std(20).over("security_id")
        ).alias("z_vs_ma20"),
        # Jegadeesh–Titman skip: 20-session return ending five sessions ago.
        (
            pl.col(PX).shift(5).over("security_id") / pl.col(PX).shift(20).over("security_id") - 1.0
        ).alias("mom_skip_5_20"),
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
        (pl.col("ret_1").abs() / dollar).rolling_mean(60).over("security_id").alias("amihud_60"),
        (pl.col("volume") / pl.col("volume").rolling_mean(20).over("security_id")).alias(
            "turnover_proxy"
        ),
        dollar.rolling_std(20).over("security_id").alias("volume_vol"),
        (
            dollar.rolling_mean(20).over("security_id")
            / dollar.rolling_mean(60).over("security_id")
        ).alias("adv_ratio_20_60"),
        pl.col(RAW_PX).log().alias("log_price"),
    )
    df = df.with_columns(_bar_characteristics())
    return _null_non_finite(df, DERIVED_BAR_COLUMNS)


# Wide PIT characteristic zoo from OHLCV only (Wave 147). Every column is a
# function of bars at or before ``event_time``. Lookbacks: 20 unless named.
DERIVED_BAR_COLUMNS: tuple[str, ...] = (
    "ret_overnight",
    "ret_open_close",
    "z_vs_ma20",
    "vol_20",
    "vol_60",
    "vol_ewma",
    "vol_parkinson",
    "vol_garman_klass",
    "vol_realized",
    "vol_of_vol",
    "amihud",
    "amihud_60",
    "rel_volume",
    "turnover_proxy",
    "volume_vol",
    "adv_ratio_20_60",
    "log_price",
    "ret_overnight_20",
    "ret_intraday_20",
    "max_ret_20",
    "min_ret_20",
    "skew_20",
    "kurt_20",
    "downside_vol_20",
    "vol_ratio_20_60",
    "high_52w_prox",
    "mom_skip_5_20",
)


def _bar_characteristics() -> list[pl.Expr]:
    """Overnight/intraday tug-of-war, MAX/MIN, realized moments, 52w high, vol term structure.

    Lou–Polk–Skouras (2019) overnight vs intraday momentum; Bali–Cakici–Whitelaw
    (2011) MAX; Amaya–Christoffersen–Jacobs–Vasquez (2015) realized skew/kurt;
    George–Hwang (2004) 52-week high. All trailing windows end at ``event_time``.
    """
    r = pl.col("log_ret_1")
    return [
        pl.col("ret_overnight").rolling_sum(20).over("security_id").alias("ret_overnight_20"),
        pl.col("ret_open_close").rolling_sum(20).over("security_id").alias("ret_intraday_20"),
        pl.col("ret_1").rolling_max(20).over("security_id").alias("max_ret_20"),
        pl.col("ret_1").rolling_min(20).over("security_id").alias("min_ret_20"),
        r.rolling_skew(20).over("security_id").alias("skew_20"),
        r.rolling_kurtosis(20).over("security_id").alias("kurt_20"),
        pl.min_horizontal(r, pl.lit(0.0))
        .pow(2)
        .rolling_mean(20)
        .over("security_id")
        .sqrt()
        .alias("downside_vol_20"),
        (pl.col("vol_20") / pl.col("vol_60")).alias("vol_ratio_20_60"),
        (pl.col(PX) / pl.col(PX).rolling_max(252).over("security_id") - 1.0).alias("high_52w_prox"),
    ]


def _null_non_finite(df: pl.DataFrame, columns: tuple[str, ...] | list[str]) -> pl.DataFrame:
    """NaN / inf in derived floats become null so ``design_matrix`` drops them honestly."""
    exprs = [
        pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c)
        for c in columns
        if c in df.columns and df.schema[c] in (pl.Float64, pl.Float32)
    ]
    return df.with_columns(exprs) if exprs else df


MARKET_RELATIVE_COLUMNS: tuple[str, ...] = ("beta_60", "idio_vol_60")
RESIDUAL_MOMENTUM_COLUMNS: tuple[str, ...] = ("idio_mom_20",)


def add_residual_momentum(df: pl.DataFrame) -> pl.DataFrame:
    """CAPM residual 20-session momentum: ``mom_20 - beta_60 * mkt_mom_20``.

    Blitz–Huij–Martens residual momentum approximated with trailing beta times
    the benchmark's own 20-session return, both ending at ``t``. Not the
    summed daily residual path.
    """
    need = {"mom_20", "beta_60", "mkt_mom_20"}
    if not need <= set(df.columns):
        return df.with_columns(pl.lit(None, dtype=pl.Float64).alias("idio_mom_20"))
    out = df.with_columns(
        (pl.col("mom_20") - pl.col("beta_60") * pl.col("mkt_mom_20")).alias("idio_mom_20")
    )
    return _null_non_finite(out, RESIDUAL_MOMENTUM_COLUMNS)


def add_market_relative_features(df: pl.DataFrame, window: int = 60) -> pl.DataFrame:
    """Trailing OLS beta vs the benchmark and residual volatility, both ending at t.

    ``beta = cov(r, m) / var(m)`` and ``idio_vol = sqrt(var(r) - cov^2 / var(m))``
    over the trailing ``window`` sessions. Requires ``mkt_ret_1`` from
    ``add_market_features``; without a benchmark both columns are null.
    """
    if "mkt_ret_1" not in df.columns or "ret_1" not in df.columns:
        return df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("beta_60"),
            pl.lit(None, dtype=pl.Float64).alias("idio_vol_60"),
        )
    out = df.sort(["security_id", "event_time"])
    cov = pl.rolling_cov(pl.col("ret_1"), pl.col("mkt_ret_1"), window_size=window).over(
        "security_id"
    )
    var_m = pl.col("mkt_ret_1").rolling_var(window).over("security_id")
    var_r = pl.col("ret_1").rolling_var(window).over("security_id")
    out = out.with_columns(
        (cov / var_m).alias("beta_60"),
        (var_r - cov.pow(2) / var_m).clip(0.0).sqrt().alias("idio_vol_60"),
    )
    return _null_non_finite(out, MARKET_RELATIVE_COLUMNS)


def add_market_features(df: pl.DataFrame, benchmark_id: str) -> pl.DataFrame:
    has_availability = "available_time" in df.columns
    eligible = decision_eligible_expr(df)
    mkt_columns = [
        "event_time",
        pl.col("ret_1").alias("mkt_ret_1"),
        pl.col("vol_20").alias("mkt_vol_20"),
        pl.col("mom_20").alias("mkt_mom_20"),
    ]
    if has_availability:
        mkt_columns.append(pl.col("available_time").alias("mkt_available_time"))
    mkt = df.filter(pl.col("security_id") == benchmark_id).select(mkt_columns)
    out = df.join(mkt, on="event_time", how="left")
    if has_availability:
        market_ok = pl.col("mkt_available_time") <= pl.col("event_time")
        market_ok = market_ok & eligible
        out = out.with_columns(
            pl.when(market_ok).then(pl.col("mkt_ret_1")).otherwise(None).alias("mkt_ret_1"),
            pl.when(market_ok).then(pl.col("mkt_vol_20")).otherwise(None).alias("mkt_vol_20"),
            pl.when(market_ok).then(pl.col("mkt_mom_20")).otherwise(None).alias("mkt_mom_20"),
        ).drop("mkt_available_time")

    # Cross-sectional aggregates are formed only from rows available at the
    # decision timestamp; output rows that are themselves late remain null.
    available = out.filter(eligible)
    cs = available.group_by("event_time").agg(
        pl.col("ret_1").std().alias("cs_dispersion"),
        (pl.col("ret_1") > 0).mean().alias("breadth"),
        pl.col("ret_1").mean().alias("cs_mean_ret"),
    )
    out = out.join(cs, on="event_time", how="left")
    aggregate_columns = ["cs_dispersion", "breadth", "cs_mean_ret"]
    out = out.with_columns(
        [
            pl.when(eligible).then(pl.col(name)).otherwise(None).alias(name)
            for name in aggregate_columns
        ]
    )
    if "sector" in out.columns:
        sec = available.group_by(["event_time", "sector"]).agg(
            pl.col("ret_1").mean().alias("sector_ret_1"),
            pl.col("mom_20").mean().alias("sector_mom_20"),
        )
        out = out.join(sec, on=["event_time", "sector"], how="left")
        out = out.with_columns(
            pl.when(eligible).then(pl.col("sector_ret_1")).otherwise(None).alias("sector_ret_1"),
            pl.when(eligible)
            .then(pl.col("mom_20") - pl.col("sector_mom_20"))
            .otherwise(None)
            .alias("sector_relative_mom_20"),
        ).drop("sector_mom_20")
    return out


# Base characteristics that receive winsor / CS-z / CS-pct / sector CS-z
# transforms at each decision date. Order is the gold column order.
CROSS_SECTIONAL_COLUMNS: tuple[str, ...] = (
    "ret_1",
    "ret_5",
    "ret_20",
    "ret_overnight",
    "ret_open_close",
    "ret_overnight_20",
    "ret_intraday_20",
    "mom_5",
    "mom_20",
    "mom_skip_5_20",
    "mom_60",
    "mom_126",
    "mom_252",
    "mom_12_1",
    "reversal_1",
    "z_vs_ma20",
    "high_52w_prox",
    "max_ret_20",
    "min_ret_20",
    "skew_20",
    "kurt_20",
    "vol_20",
    "vol_60",
    "vol_ewma",
    "vol_parkinson",
    "vol_garman_klass",
    "vol_of_vol",
    "downside_vol_20",
    "vol_ratio_20_60",
    "beta_60",
    "idio_vol_60",
    "idio_mom_20",
    "amihud",
    "amihud_60",
    "adv",
    "dollar_volume",
    "rel_volume",
    "turnover_proxy",
    "volume_vol",
    "adv_ratio_20_60",
    "log_price",
    "planted_signal",
)


def feature_catalog() -> list[FeatureMetadata]:
    cols = [
        ("ret_1", 1, "returns"),
        ("ret_5", 5, "returns"),
        ("ret_20", 20, "returns"),
        ("log_ret_1", 1, "returns"),
        ("ret_overnight", 1, "returns"),
        ("ret_open_close", 1, "returns"),
        ("ret_overnight_20", 20, "momentum"),
        ("ret_intraday_20", 20, "momentum"),
        ("mom_5", 5, "momentum"),
        ("mom_20", 20, "momentum"),
        ("mom_skip_5_20", 20, "momentum"),
        ("mom_60", 60, "momentum"),
        ("mom_126", 126, "momentum"),
        ("mom_252", 252, "momentum"),
        ("mom_12_1", 252, "momentum"),
        ("high_52w_prox", 252, "momentum"),
        ("reversal_1", 1, "reversal"),
        ("z_vs_ma20", 20, "reversal"),
        ("max_ret_20", 20, "lottery"),
        ("min_ret_20", 20, "lottery"),
        ("skew_20", 20, "moments"),
        ("kurt_20", 20, "moments"),
        ("vol_20", 20, "volatility"),
        ("vol_60", 60, "volatility"),
        ("vol_ewma", 20, "volatility"),
        ("vol_parkinson", 20, "volatility"),
        ("vol_garman_klass", 20, "volatility"),
        ("vol_of_vol", 40, "volatility"),
        ("downside_vol_20", 20, "volatility"),
        ("vol_ratio_20_60", 60, "volatility"),
        ("beta_60", 60, "market"),
        ("idio_vol_60", 60, "market"),
        ("idio_mom_20", 20, "momentum"),
        ("amihud", 20, "liquidity"),
        ("amihud_60", 60, "liquidity"),
        ("adv", 20, "liquidity"),
        ("dollar_volume", 1, "liquidity"),
        ("rel_volume", 20, "liquidity"),
        ("turnover_proxy", 20, "liquidity"),
        ("volume_vol", 20, "liquidity"),
        ("adv_ratio_20_60", 60, "liquidity"),
        ("log_price", 1, "price"),
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
    membership: pl.DataFrame | None = None,
) -> pl.DataFrame:
    df = compute_base_features(bars, config)
    if membership is not None:
        if membership.is_empty() and bars.height:
            raise PointInTimeError(
                "universe membership is empty; refusing unfiltered feature panel"
            )
        df = attach_membership_flag(df, membership)
    df = add_market_features(df, config.data.benchmark_id)
    df = add_market_relative_features(df)
    df = add_residual_momentum(df)
    names = [c for c in CROSS_SECTIONAL_COLUMNS if c in df.columns]
    sector = "sector" if "sector" in df.columns else None
    df = apply_cross_sectional(df, names, config.features.winsor_p, sector=sector)
    if "_in_universe" in df.columns:
        df = df.filter(pl.col("_in_universe").fill_null(False)).drop("_in_universe")
        if df.is_empty():
            raise PointInTimeError("membership filter removed every feature row")
    df = df.with_columns(
        pl.col("available_time").max().over("event_time").alias("max_source_available_time"),
        pl.col("event_time").alias("decision_time"),
        pl.lit(FEATURE_SET_VERSION).alias("feature_set_version"),
    )
    # Validate the complete feature panel before it can become training data.
    # The validator compares each row's source availability with its own
    # decision timestamp, so mixed as-of panels cannot silently leak later data.
    as_of = decision_time or cast(datetime, df["event_time"].min())
    validate_feature_frame(df, as_of)
    return df
