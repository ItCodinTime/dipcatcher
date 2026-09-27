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
    """Winsorize, robust-z, and percentile-rank each column inside ``event_time``.

    Statistics are group-by reductions joined back onto the rows. That matches
    the per-column ``over`` window (nulls stay out of the cross-section; ranks
    are average ranks; MAD falls back to the group standard deviation) without
    replanning one window query per column.
    """
    if not columns:
        return df
    lo, hi = winsor_p, 1.0 - winsor_p
    eligible = decision_eligible_expr(df)
    use_sector = sector is not None and sector in df.columns
    src = [f"_cs_src_{col}" for col in columns]
    work = df.with_columns(
        [
            pl.when(eligible).then(pl.col(col)).otherwise(None).alias(name)
            for col, name in zip(columns, src, strict=True)
        ]
    )
    reductions: list[pl.Expr] = []
    for col, name in zip(columns, src, strict=True):
        reductions.extend(
            [
                pl.col(name).quantile(lo).alias(f"_cs_lo_{col}"),
                pl.col(name).quantile(hi).alias(f"_cs_hi_{col}"),
                pl.col(name).count().alias(f"_cs_n_{col}"),
            ]
        )
    work = work.join(work.group_by("event_time").agg(reductions), on="event_time", how="left")
    clip = [f"_cs_clip_{col}" for col in columns]
    work = work.with_columns(
        [
            pl.col(name).clip(pl.col(f"_cs_lo_{col}"), pl.col(f"_cs_hi_{col}")).alias(clipped)
            for col, name, clipped in zip(columns, src, clip, strict=True)
        ]
    )
    location: list[pl.Expr] = []
    for col, clipped in zip(columns, clip, strict=True):
        location.extend(
            [
                pl.col(clipped).median().alias(f"_cs_med_{col}"),
                pl.col(clipped).std().alias(f"_cs_sd_{col}"),
            ]
        )
    work = work.join(work.group_by("event_time").agg(location), on="event_time", how="left")
    dev = [f"_cs_dev_{col}" for col in columns]
    work = work.with_columns(
        [
            (pl.col(clipped) - pl.col(f"_cs_med_{col}")).abs().alias(name)
            for col, clipped, name in zip(columns, clip, dev, strict=True)
        ]
    )
    work = work.join(
        work.group_by("event_time").agg(
            [
                pl.col(name).median().alias(f"_cs_mad_{col}")
                for col, name in zip(columns, dev, strict=True)
            ]
        ),
        on="event_time",
        how="left",
    )
    work = work.with_columns(
        [
            pl.col(name).rank("average").over("event_time").alias(f"_cs_rk_{col}")
            for col, name in zip(columns, src, strict=True)
        ]
    )
    if use_sector:
        assert sector is not None
        # Null sector labels are their own group. A left join does not match
        # null keys, so the key is (is_null, filled label) instead of the raw
        # column the window expression grouped on.
        work = work.with_columns(
            pl.col(sector).is_null().alias("_cs_sec_null"),
            pl.col(sector).cast(pl.Utf8).fill_null("").alias("_cs_sec"),
        )
        sec_keys = ["event_time", "_cs_sec_null", "_cs_sec"]
        sec_loc: list[pl.Expr] = []
        for col, clipped in zip(columns, clip, strict=True):
            sec_loc.extend(
                [
                    pl.col(clipped).median().alias(f"_cs_smed_{col}"),
                    pl.col(clipped).std().alias(f"_cs_ssd_{col}"),
                ]
            )
        work = work.join(work.group_by(sec_keys).agg(sec_loc), on=sec_keys, how="left")
        sdev = [f"_cs_sdev_{col}" for col in columns]
        work = work.with_columns(
            [
                (pl.col(clipped) - pl.col(f"_cs_smed_{col}")).abs().alias(name)
                for col, clipped, name in zip(columns, clip, sdev, strict=True)
            ]
        )
        work = work.join(
            work.group_by(sec_keys).agg(
                [
                    pl.col(name).median().alias(f"_cs_smad_{col}")
                    for col, name in zip(columns, sdev, strict=True)
                ]
            ),
            on=sec_keys,
            how="left",
        )
    final: list[pl.Expr] = []
    for col, clipped in zip(columns, clip, strict=True):
        z = _robust_z_from_parts(
            pl.col(clipped),
            pl.col(f"_cs_med_{col}"),
            pl.col(f"_cs_mad_{col}"),
            pl.col(f"_cs_sd_{col}"),
            eligible,
        )
        final.extend(
            [
                pl.when(eligible).then(pl.col(clipped)).otherwise(None).alias(f"winsor_{col}"),
                z.alias(f"cs_z_{col}"),
                pl.when(eligible)
                .then((pl.col(f"_cs_rk_{col}") - 0.5) / pl.col(f"_cs_n_{col}"))
                .otherwise(None)
                .alias(f"cs_pct_{col}"),
            ]
        )
        if use_sector:
            final.append(
                _robust_z_from_parts(
                    pl.col(clipped),
                    pl.col(f"_cs_smed_{col}"),
                    pl.col(f"_cs_smad_{col}"),
                    pl.col(f"_cs_ssd_{col}"),
                    eligible,
                ).alias(f"cs_z_sector_{col}")
            )
    work = work.with_columns(final)
    drop = [name for name in work.columns if name.startswith("_cs_")]
    return work.drop(drop)


_SCALE_FLOOR = 1e-12


def _robust_z_from_parts(
    value: pl.Expr,
    med: pl.Expr,
    mad: pl.Expr,
    sd: pl.Expr,
    eligible: pl.Expr,
) -> pl.Expr:
    """Median / MAD z-score with a standard-deviation fallback when MAD is 0.

    More than half of a small cross-section can share one value (names at
    their 52-week high, a zero-volume day), which makes MAD exactly 0 and
    ``(x - med) / (1.4826 * MAD + 1e-12)`` explode to ~1e10 for every other
    name. Rousseeuw–Croux style fallback: use the group standard deviation
    when MAD is degenerate, and 0 (no dispersion, no information) when both
    are degenerate. ``med`` / ``mad`` / ``sd`` are already within-group, so
    PIT is unchanged from the window form of this score.
    """
    scale = pl.when(mad > _SCALE_FLOOR).then(1.4826 * mad).otherwise(sd)
    z = pl.when(scale > _SCALE_FLOOR).then((value - med) / scale).otherwise(0.0)
    return pl.when(eligible).then(z).otherwise(None)
