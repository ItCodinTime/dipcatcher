"""Time-aware ticker → security_id mapping."""

from __future__ import annotations

from datetime import datetime

import polars as pl


class FrameSecurityMaster:
    def __init__(self, master: pl.DataFrame) -> None:
        self._m = master

    def asof(self, when: datetime, ticker: str) -> str | None:
        hit = self._m.filter(
            (pl.col("ticker") == ticker)
            & (pl.col("valid_from") <= when)
            & (pl.col("valid_to").is_null() | (pl.col("valid_to") > when))
        )
        if hit.is_empty():
            return None
        return str(hit["security_id"][0])

    def record(self, security_id: str, when: datetime) -> dict[str, str] | None:
        hit = self._m.filter(
            (pl.col("security_id") == security_id)
            & (pl.col("valid_from") <= when)
            & (pl.col("valid_to").is_null() | (pl.col("valid_to") > when))
        )
        if hit.is_empty():
            return None
        row = hit.row(0, named=True)
        return {k: str(v) for k, v in row.items()}

    def frame(self) -> pl.DataFrame:
        return self._m
