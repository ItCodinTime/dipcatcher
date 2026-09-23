"""Cross-sectional transforms computed independently at each timestamp."""

from __future__ import annotations

import polars as pl


def apply_cross_sectional(
    df: pl.DataFrame,
    columns: list[str],
    winsor_p: float,
    sector: str | None = None,
) -> pl.DataFrame:
    out = df
    lo, hi = winsor_p, 1.0 - winsor_p
    eligible = (
        pl.col("available_time") <= pl.col("event_time")
        if "available_time" in df.columns
        else pl.lit(True)
    )
    # A PIT universe flag is an additional eligibility gate.  Keep the full
    # panel available to callers so rolling/history calculations are preserved,
    # while excluding non-members from every cross-sectional statistic.
    if "_in_universe" in df.columns:
        eligible = eligible & pl.col("_in_universe")
    for col in columns:
        g = pl.col(col)
        # Late-arriving observations must not influence the cross-section at
        # their event time.  Nulling them before every aggregate also keeps
        # the denominator and rank universe PIT-correct.
        source = pl.when(eligible).then(g).otherwise(None)
        q_lo = source.quantile(lo).over("event_time")
        q_hi = source.quantile(hi).over("event_time")
        clipped_base = source.clip(q_lo, q_hi)
        clipped = pl.when(eligible).then(clipped_base).otherwise(None)
        med = clipped_base.median().over("event_time")
        mad = (clipped_base - med).abs().median().over("event_time")
        robust_z = (
            pl.when(eligible).then((clipped_base - med) / (1.4826 * mad + 1e-12)).otherwise(None)
        )
        rank = source.rank("average").over("event_time")
        n = source.count().over("event_time")
        pct = pl.when(eligible).then((rank - 0.5) / n).otherwise(None)
        out = out.with_columns(
            clipped.alias(f"winsor_{col}"),
            robust_z.alias(f"cs_z_{col}"),
            pct.alias(f"cs_pct_{col}"),
        )
        if sector is not None and sector in out.columns:
            med_s = clipped_base.median().over(["event_time", sector])
            mad_s = (clipped_base - med_s).abs().median().over(["event_time", sector])
            out = out.with_columns(
                pl.when(eligible)
                .then((clipped_base - med_s) / (1.4826 * mad_s + 1e-12))
                .otherwise(None)
                .alias(f"cs_z_sector_{col}")
            )
    return out
