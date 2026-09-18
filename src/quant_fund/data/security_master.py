"""Time-aware ticker → security_id mapping."""

from __future__ import annotations

from datetime import datetime

import polars as pl


class FrameSecurityMaster:
    def __init__(self, master: pl.DataFrame) -> None:
        self._m = master

    def asof(self, when: datetime, ticker: str) -> str | None:
        hit = (
            self._m.filter(
                (pl.col("ticker") == ticker)
                & (pl.col("valid_from") <= when)
                & (pl.col("valid_to").is_null() | (pl.col("valid_to") > when))
            )
            # Overlapping vintages resolve to the most recent valid_from rather
            # than to input row order.
            .sort("valid_from")
        )
        if hit.is_empty():
            return None
        return str(hit["security_id"][-1])

    def record(self, security_id: str, when: datetime) -> dict[str, object] | None:
        hit = self._m.filter(
            (pl.col("security_id") == security_id)
            & (pl.col("valid_from") <= when)
            & (pl.col("valid_to").is_null() | (pl.col("valid_to") > when))
        ).sort("valid_from")
        if hit.is_empty():
            return None
        # Preserve raw values: str(None) would turn an open-ended valid_to into
        # the string "None" for consumers comparing against datetimes.
        return dict(hit.row(-1, named=True))

    def frame(self) -> pl.DataFrame:
        return self._m
