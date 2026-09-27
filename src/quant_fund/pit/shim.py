"""Migration shim wrapping Lake + the 32 rogue parquet reads (§4.4, A2 F8).

Staged migration: the shim exists day 1; LH009 reports the rogue sites as
warnings; the CI gate flips LH009 to error only after the call-site migration
PR lands. The shim NEVER rewrites the underlying parquet.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from quant_fund.pit.corrections import (
    EVENT_TIME_COL,
    KNOWN_AT_COL,
    SECURITY_ID_COL,
    RestatementPolicy,
    select_asof,
)
from quant_fund.pit.frame import PitFrame, VaultUnavailableError
from quant_fund.schemas.errors import PointInTimeError

if TYPE_CHECKING:
    from quant_fund.data.lake import Lake


def _to_pit_frame(frame: pl.DataFrame, *, dataset: str, t: datetime) -> PitFrame:
    """Map legacy PIT columns onto the vault contract and apply the filter."""
    if t.tzinfo is None or t.tzinfo.utcoffset(t) is None:
        raise PointInTimeError(f"{dataset}: asof timestamp must be timezone-aware (UTC)")
    t = t.astimezone(timezone.utc)
    if EVENT_TIME_COL not in frame.columns:
        raise PointInTimeError(
            f"{dataset}: frame lacks 'event_time' — PIT observability cannot be "
            "established; refusing the read (fail-closed)"
        )
    if KNOWN_AT_COL not in frame.columns:
        # Legacy column set (data/point_in_time.py PIT_COLS): available_time
        # is the publication timestamp; map it onto the vault's known_at.
        if "available_time" not in frame.columns:
            raise PointInTimeError(
                f"{dataset}: frame lacks 'available_time'/'known_at' — PIT "
                "observability cannot be established; refusing the read (fail-closed)"
            )
        frame = frame.with_columns(pl.col("available_time").alias(KNOWN_AT_COL))
    dtype = frame.schema[KNOWN_AT_COL]
    if not isinstance(dtype, pl.Datetime) or dtype.time_zone is None:
        raise PointInTimeError(
            f"{dataset}: {KNOWN_AT_COL} must be timezone-aware Datetime, got {dtype} — "
            "fail-closed"
        )
    key_cols = (
        [SECURITY_ID_COL, EVENT_TIME_COL]
        if SECURITY_ID_COL in frame.columns
        else [EVENT_TIME_COL]
    )
    sliced = select_asof(
        frame.lazy(),
        t,
        key_cols=key_cols,
        policy=RestatementPolicy.LATEST_KNOWN,
    )
    if sliced.height == 0:
        raise VaultUnavailableError(
            f"{dataset}: no version with known_at <= {t.isoformat()} exists"
        )
    pit_frame = PitFrame.build(sliced, dataset=dataset, asof=t)
    pit_frame.validate(t)
    return pit_frame


def lake_asof(lake: Lake, rel: str, t: datetime) -> PitFrame:
    """Wraps Lake.read_parquet: loads, maps available_time -> known_at if the
    legacy column set is present (data/point_in_time.py PIT_COLS), then applies
    the vault filter. Frames lacking PIT columns raise PointInTimeError
    (fail-closed — this is the whole point)."""
    return _to_pit_frame(lake.read_parquet(rel), dataset=rel, t=t)


def guarded_read_parquet(path: Path, t: datetime, *, dataset: str) -> PitFrame:
    """Replacement for the 32 rogue pl.read_parquet sites (A2 F8)."""
    return _to_pit_frame(pl.read_parquet(Path(path)), dataset=dataset, t=t)
