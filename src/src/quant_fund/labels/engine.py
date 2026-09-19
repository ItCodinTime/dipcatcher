"""Horizon labels. Forward-looking; never used as features at t."""

from __future__ import annotations

import polars as pl

from quant_fund.config.models import AppConfig

PX = "close_total_return"


_LABEL_REQUIRED = ("security_id", "event_time", PX)


def build_labels(bars: pl.DataFrame, config: AppConfig) -> pl.DataFrame:
    if bars.height == 0:
        raise ValueError("bars must be non-empty for build_labels")
    missing = [c for c in _LABEL_REQUIRED if c not in bars.columns]
    if missing:
        raise ValueError(f"bars missing required OHLCV columns: {missing}")
    df = bars.sort(["security_id", "event_time"])
    mkt_id = config.data.benchmark_id
    # Compute benchmark forwards on the benchmark's own observed calendar before
    # joining them to securities.  Shifting after a panel join would make a
    # missing/asynchronous security row change which benchmark observation is
    # considered ``t+h``.
    mkt = df.filter(pl.col("security_id") == mkt_id).select("event_time", PX).sort("event_time")
    for h, name in zip(config.horizons.bars, config.horizons.names, strict=True):
        fut = pl.col(PX).shift(-h).over("security_id")
        label_end = pl.col("event_time").shift(-h).over("security_id")
        df = df.with_columns(
            # Labels retain observed-row horizons.  Persist the exact endpoint
            # because sparse securities do not share the benchmark/session grid.
            label_end.alias(f"label_end_time_{h}"),
            (fut / pl.col(PX) - 1.0).alias(f"future_return_{h}"),
            (fut / pl.col(PX)).log().alias(f"future_log_return_{h}"),
        )
        mkt_fwd = mkt.with_columns(
            (pl.col(PX).shift(-h) / pl.col(PX) - 1.0).alias(f"_mkt_fwd_{h}")
        ).select("event_time", f"_mkt_fwd_{h}")
        df = df.join(mkt_fwd, on="event_time", how="left")
        df = df.with_columns(
            (pl.col(f"future_return_{h}") - pl.col(f"_mkt_fwd_{h}")).alias(
                f"future_excess_return_{h}"
            )
        )
        df = df.with_columns(
            (
                pl.col(f"future_return_{h}")
                - pl.col(f"future_return_{h}").mean().over("event_time")
            ).alias(f"future_idio_return_{h}")
        )
        # Realized volatility is the square root of forward squared log returns.
        # This remains defined for a one-bar horizon (unlike sample std, whose
        # one-observation value is null) and matches the usual high-frequency RV
        # aggregation convention.
        r1 = (pl.col(PX) / pl.col(PX).shift(1).over("security_id")).log()
        df = df.with_columns(r1.alias("_r1"))
        fwd_r = [pl.col("_r1").shift(-k).over("security_id") for k in range(1, h + 1)]
        rv_path = pl.concat_list(fwd_r)
        df = df.with_columns(
            rv_path.list.eval(pl.element() ** 2)
            .list.sum()
            .sqrt()
            .alias(f"future_realized_vol_{h}"),
        )
        df = df.with_columns(
            (pl.col(f"future_realized_vol_{h}") ** 2).alias(f"future_realized_var_{h}"),
        )
        # Maximum peak-to-trough drawdown over the forward TR-wealth path.
        # Include the origin (wealth=1) so an immediate decline is represented,
        # then measure each future wealth point against the running peak.
        fwd = [pl.col(PX).shift(-k).over("security_id") / pl.col(PX) for k in range(1, h + 1)]
        if fwd:
            path = pl.concat_list([pl.lit(1.0), *fwd])
            drawdowns = path.list.eval(pl.element() / pl.element().cum_max() - 1.0)
            df = df.with_columns(drawdowns.list.min().alias(f"future_max_drawdown_{h}"))
            df = df.with_columns(
                (pl.col(f"future_max_drawdown_{h}") < -0.05)
                .cast(pl.Float64)
                .alias(f"future_tail_event_{h}")
            )
        _ = name
    if "sector" in df.columns:
        h = config.horizons.bars[0]
        sec = df.group_by(["event_time", "sector"]).agg(
            pl.col(f"future_return_{h}").mean().alias(f"_sec_fwd_{h}")
        )
        df = df.join(sec, on=["event_time", "sector"], how="left")
        df = df.with_columns(
            (pl.col(f"future_return_{h}") - pl.col(f"_sec_fwd_{h}")).alias(
                f"future_sector_relative_return_{h}"
            )
        )
    drop = [c for c in df.columns if c.startswith("_")]
    return df.drop(drop)
