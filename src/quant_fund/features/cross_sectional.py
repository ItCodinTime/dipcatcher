"""Cross-sectional transforms computed independently at each timestamp."""

from __future__ import annotations

import polars as pl

_MEMBERSHIP_FLAG = "_in_universe"


def decision_eligible_expr(df: pl.DataFrame) -> pl.Expr:
    """Rows allowed to enter a decision-time cross-section.

    Late ``available_time`` is excluded (Wave 94). When ``_in_universe`` is
    stamped, names outside the PIT membership panel are also excluded so
    ineligible ADV/listing rows cannot move ranks or market aggregates.
    """
    eligible = (
        pl.col("available_time") <= pl.col("event_time")
        if "available_time" in df.columns
        else pl.lit(True)
    )
    if _MEMBERSHIP_FLAG in df.columns:
        eligible = eligible & pl.col(_MEMBERSHIP_FLAG).fill_null(False)
    return eligible


def apply_cross_sectional(
    df: pl.DataFrame,
    columns: list[str],
    winsor_p: float,
    sector: str | None = None,
) -> pl.DataFrame:
    out = df
    lo, hi = winsor_p, 1.0 - winsor_p
    eligible = decision_eligible_expr(df)
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
        robust_z = _robust_z(clipped_base, eligible, ["event_time"])
        rank = source.rank("average").over("event_time")
        n = source.count().over("event_time")
        pct = pl.when(eligible).then((rank - 0.5) / n).otherwise(None)
        out = out.with_columns(
            clipped.alias(f"winsor_{col}"),
            robust_z.alias(f"cs_z_{col}"),
            pct.alias(f"cs_pct_{col}"),
        )
        if sector is not None and sector in out.columns:
            out = out.with_columns(
                _robust_z(clipped_base, eligible, ["event_time", sector]).alias(
                    f"cs_z_sector_{col}"
                )
            )
    return out


_SCALE_FLOOR = 1e-12


def _robust_z(value: pl.Expr, eligible: pl.Expr, keys: list[str]) -> pl.Expr:
    """Median / MAD z-score with a standard-deviation fallback when MAD is 0.

    More than half of a small cross-section can share one value (names at
    their 52-week high, a zero-volume day), which makes MAD exactly 0 and
    ``(x - med) / (1.4826 * MAD + 1e-12)`` explode to ~1e10 for every other
    name. Rousseeuw–Croux style fallback: use the group standard deviation
    when MAD is degenerate, and 0 (no dispersion, no information) when both
    are degenerate. Everything is within-group, so PIT is unchanged.
    """
    med = value.median().over(keys)
    mad = (value - med).abs().median().over(keys)
    sd = value.std().over(keys)
    scale = pl.when(mad > _SCALE_FLOOR).then(1.4826 * mad).otherwise(sd)
    z = pl.when(scale > _SCALE_FLOOR).then((value - med) / scale).otherwise(0.0)
    return pl.when(eligible).then(z).otherwise(None)
