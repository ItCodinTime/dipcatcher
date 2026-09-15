"""CSV / Parquet file adapter. No network."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl


class ParquetMarketProvider:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _load(self, name: str) -> pl.DataFrame:
        pq = self.root / f"{name}.parquet"
        csv = self.root / f"{name}.csv"
        if pq.exists():
            return pl.read_parquet(pq)
        if csv.exists():
            return pl.read_csv(csv, try_parse_dates=True)
        return pl.DataFrame()

    def get_bars(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        security_ids: list[str] | None = None,
    ) -> pl.DataFrame:
        df = self._load("bars")
        if df.is_empty():
            return df
        if start is not None:
            df = df.filter(pl.col("event_time") >= start)
        if end is not None:
            df = df.filter(pl.col("event_time") <= end)
        if security_ids is not None:
            df = df.filter(pl.col("security_id").is_in(security_ids))
        return df

    def get_corporate_actions(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        df = self._load("corporate_actions")
        if df.is_empty():
            return df
        if start is not None:
            df = df.filter(pl.col("event_time") >= start)
        if end is not None:
            df = df.filter(pl.col("event_time") <= end)
        return df

    def get_security_master(self) -> pl.DataFrame:
        return self._load("security_master")
