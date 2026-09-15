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
    for col in columns:
        g = pl.col(col)
        q_lo = g.quantile(lo).over("event_time")
        q_hi = g.quantile(hi).over("event_time")
        clipped = g.clip(q_lo, q_hi)
        med = clipped.median().over("event_time")
        mad = (clipped - med).abs().median().over("event_time")
        robust_z = (clipped - med) / (1.4826 * mad + 1e-12)
        rank = g.rank("average").over("event_time")
        n = pl.len().over("event_time")
        pct = (rank - 0.5) / n
        out = out.with_columns(
            clipped.alias(f"winsor_{col}"),
            robust_z.alias(f"cs_z_{col}"),
            pct.alias(f"cs_pct_{col}"),
        )
        if sector is not None and sector in out.columns:
            med_s = clipped.median().over(["event_time", sector])
            mad_s = (clipped - med_s).abs().median().over(["event_time", sector])
            out = out.with_columns(
                ((clipped - med_s) / (1.4826 * mad_s + 1e-12)).alias(f"cs_z_sector_{col}")
            )
    return out
