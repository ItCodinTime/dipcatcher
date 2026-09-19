"""Time-aware ticker → security_id mapping."""

from __future__ import annotations

from datetime import datetime

import polars as pl

from quant_fund.schemas.errors import PointInTimeError

_ATTR_COLS = ("sector", "industry", "exchange")


def snapshot_asof(master: pl.DataFrame, when: datetime) -> pl.DataFrame:
    """Return at most one observable vintage per security_id at ``when``.

    Frames without ``valid_from`` keep the input rows (legacy compatibility).
    When ``available_time`` is present, null or future availability is excluded
    rather than treated as observable. Overlapping vintages resolve to the most
    recent ``valid_from``. Duplicate ``security_id`` rows that cannot be ordered
    fail closed.
    """
    if master.is_empty():
        return master
    visible = master
    if "valid_from" in visible.columns:
        visible = visible.filter(pl.col("valid_from") <= when)
        if "valid_to" in visible.columns:
            visible = visible.filter(pl.col("valid_to").is_null() | (pl.col("valid_to") > when))
    if "available_time" in visible.columns:
        visible = visible.filter(pl.col("available_time") <= when)
    if visible.is_empty():
        return visible
    if "valid_from" in visible.columns:
        return visible.sort("valid_from").unique(
            subset=["security_id"], keep="last", maintain_order=True
        )
    if visible.select("security_id").is_duplicated().any():
        raise PointInTimeError("security master as-of snapshot contains duplicate security_id rows")
    return visible


def attach_master_attributes(bars: pl.DataFrame, master: pl.DataFrame) -> pl.DataFrame:
    """Left-attach PIT-valid sector/industry/exchange without exploding bars.

    Legacy masters without ``valid_from`` still join on ``security_id`` but fail
    closed on duplicate identity rows. Vintaged masters keep, for each bar, the
    latest mapping that was economically valid and available at ``event_time``.
    """
    if bars.is_empty() or master.is_empty():
        return bars
    attr_cols = [name for name in _ATTR_COLS if name in master.columns]
    if not attr_cols or "security_id" not in master.columns:
        return bars
    if "valid_from" not in master.columns:
        attrs = master.select(["security_id", *attr_cols])
        if attrs.select("security_id").is_duplicated().any():
            raise PointInTimeError("security master contains duplicate security_id rows")
        return bars.join(attrs, on="security_id", how="left")
    if "event_time" not in bars.columns:
        raise PointInTimeError("bars missing event_time for security-master attach")

    extra = [name for name in ("valid_to", "available_time") if name in master.columns]
    rename: dict[str, str] = {
        name: f"{name}_master"
        for name in ("valid_from", "valid_to", "available_time")
        if name in master.columns
    }
    right = master.select(["security_id", "valid_from", *attr_cols, *extra]).rename(rename)
    valid_from_col = rename["valid_from"]
    joined = bars.join(right, on="security_id", how="left")
    eligible = pl.col(valid_from_col).is_not_null() & (
        pl.col(valid_from_col) <= pl.col("event_time")
    )
    if "valid_to" in extra:
        valid_to_col = rename["valid_to"]
        eligible = eligible & (
            pl.col(valid_to_col).is_null() | (pl.col(valid_to_col) > pl.col("event_time"))
        )
    if "available_time" in extra:
        avail_col = rename["available_time"]
        eligible = (
            eligible & pl.col(avail_col).is_not_null() & (pl.col(avail_col) <= pl.col("event_time"))
        )
    chosen = (
        joined.filter(eligible)
        .sort(valid_from_col)
        .unique(subset=["security_id", "event_time"], keep="last", maintain_order=True)
        .select(["security_id", "event_time", *attr_cols])
    )
    return bars.join(chosen, on=["security_id", "event_time"], how="left")


class FrameSecurityMaster:
    def __init__(self, master: pl.DataFrame) -> None:
        self._m = master

    def asof(self, when: datetime, ticker: str) -> str | None:
        hit = self._m.filter(
            (pl.col("ticker") == ticker)
            & (pl.col("valid_from") <= when)
            & (pl.col("valid_to").is_null() | (pl.col("valid_to") > when))
        )
        # Late-arriving restatements cannot rewrite pre-availability identity.
        # Frames without available_time retain the legacy valid-window lookup.
        if "available_time" in hit.columns:
            hit = hit.filter(pl.col("available_time") <= when)
        hit = hit.sort("valid_from")
        if hit.is_empty():
            return None
        return str(hit["security_id"][-1])

    def record(self, security_id: str, when: datetime) -> dict[str, object] | None:
        hit = self._m.filter(
            (pl.col("security_id") == security_id)
            & (pl.col("valid_from") <= when)
            & (pl.col("valid_to").is_null() | (pl.col("valid_to") > when))
        )
        if "available_time" in hit.columns:
            hit = hit.filter(pl.col("available_time") <= when)
        hit = hit.sort("valid_from")
        if hit.is_empty():
            return None
        # Preserve raw values: str(None) would turn an open-ended valid_to into
        # the string "None" for consumers comparing against datetimes.
        return dict(hit.row(-1, named=True))

    def frame(self) -> pl.DataFrame:
        return self._m
