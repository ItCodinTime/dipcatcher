"""CSV / Parquet file adapter. No network."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from quant_fund.data.point_in_time import require_pit_columns
from quant_fund.schemas.errors import PointInTimeError

_BAR_COLUMNS = {
    "event_time",
    "security_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


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
        missing = sorted(_BAR_COLUMNS - set(df.columns))
        if missing:
            raise PointInTimeError(f"bars adapter missing required columns: {missing}")
        require_pit_columns(df)
        temporal = ("event_time", "available_time", "ingested_time")
        if any(df.schema[name] not in (pl.Date, pl.Datetime) for name in temporal):
            raise PointInTimeError("bars adapter PIT timestamps must be Date or Datetime")
        invalid = df.filter(
            pl.any_horizontal(
                pl.col("event_time").is_null(),
                pl.col("available_time").is_null(),
                pl.col("ingested_time").is_null(),
                # Data cannot be available before its own event (PIT contract,
                # mirrors the book-panel validator).
                pl.col("event_time") > pl.col("available_time"),
                pl.col("available_time") > pl.col("ingested_time"),
                pl.col("security_id").is_null(),
                pl.col("source").is_null()
                | (pl.col("source").cast(pl.String).str.strip_chars() == ""),
            )
        )
        if invalid.height:
            raise PointInTimeError("bars adapter contains null, blank, or impossible PIT fields")
        bad_prices = df.filter(
            pl.any_horizontal(
                *[
                    (~pl.col(name).is_finite()) | (pl.col(name) <= 0)
                    for name in ("open", "high", "low", "close")
                ],
                pl.col("high") < pl.col("low"),
                pl.col("open") > pl.col("high"),
                pl.col("close") > pl.col("high"),
                pl.col("open") < pl.col("low"),
                pl.col("close") < pl.col("low"),
                pl.col("volume").is_null()
                | (~pl.col("volume").is_finite())
                | (pl.col("volume") < 0),
            )
        )
        if bad_prices.height:
            raise PointInTimeError("bars adapter contains invalid OHLCV values")
        if df.select(["event_time", "security_id"]).is_duplicated().any():
            raise PointInTimeError("bars adapter contains duplicate event_time/security_id rows")
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
