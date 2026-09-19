"""CSV / Parquet file adapter. No network."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from quant_fund.data.corporate_actions import require_valid_ticker_changes
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
_CA_COLUMNS = {
    "event_time",
    "security_id",
    "action_type",
}
_SM_COLUMNS = {
    "security_id",
    "ticker",
    "valid_from",
    "valid_to",
    "available_time",
    "ingested_time",
    "source",
    "revision_id",
}
_ALLOWED_ACTION_TYPES = frozenset(
    {
        "split",
        "cash_dividend",
        "special_dividend",
        "delist",
        "ticker_change",
    }
)
_DIVIDEND_ACTION_TYPES = frozenset({"cash_dividend", "special_dividend"})


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
        missing = sorted(_CA_COLUMNS - set(df.columns))
        if missing:
            raise PointInTimeError(f"corporate actions missing required columns: {missing}")
        require_pit_columns(df)
        temporal = ("event_time", "available_time", "ingested_time")
        if any(df.schema[name] not in (pl.Date, pl.Datetime) for name in temporal):
            raise PointInTimeError("corporate actions PIT timestamps must be Date or Datetime")
        invalid = df.filter(
            pl.any_horizontal(
                pl.col("event_time").is_null(),
                pl.col("available_time").is_null(),
                pl.col("ingested_time").is_null(),
                pl.col("event_time") > pl.col("available_time"),
                pl.col("available_time") > pl.col("ingested_time"),
                pl.col("security_id").is_null(),
                pl.col("source").is_null()
                | (pl.col("source").cast(pl.String).str.strip_chars() == ""),
            )
        )
        if invalid.height:
            raise PointInTimeError(
                "corporate actions contain null, blank, or impossible PIT fields"
            )
        df = df.with_columns(pl.col("action_type").cast(pl.String).str.strip_chars())
        invalid_type = df.filter(pl.col("action_type").is_null() | (pl.col("action_type") == ""))
        if invalid_type.height:
            raise PointInTimeError("corporate actions contain null or blank action_type")
        unknown = df.filter(~pl.col("action_type").is_in(sorted(_ALLOWED_ACTION_TYPES)))
        if unknown.height:
            raise PointInTimeError("corporate actions contain unknown action_type")
        splits = df.filter(pl.col("action_type") == "split")
        if splits.height:
            if "factor" not in df.columns:
                raise PointInTimeError("split actions missing factor")
            factor = pl.col("factor").cast(pl.Float64, strict=False)
            bad_factor = splits.filter(factor.is_null() | (~factor.is_finite()) | (factor <= 0))
            if bad_factor.height:
                raise PointInTimeError("split actions contain non-positive or non-finite factors")
        dividends = df.filter(pl.col("action_type").is_in(sorted(_DIVIDEND_ACTION_TYPES)))
        if dividends.height:
            if "amount" not in df.columns:
                raise PointInTimeError("dividend actions missing amount")
            amount = pl.col("amount").cast(pl.Float64, strict=False)
            bad_amount = dividends.filter(amount.is_null() | (~amount.is_finite()) | (amount < 0))
            if bad_amount.height:
                raise PointInTimeError("dividend actions contain negative or non-finite amounts")
        require_valid_ticker_changes(df)
        if df.select(["security_id", "event_time", "action_type"]).is_duplicated().any():
            raise PointInTimeError(
                "corporate actions contain duplicate security_id/event_time/action_type rows"
            )
        if start is not None:
            df = df.filter(pl.col("event_time") >= start)
        if end is not None:
            df = df.filter(pl.col("event_time") <= end)
        return df

    def get_security_master(self) -> pl.DataFrame:
        df = self._load("security_master")
        if df.is_empty():
            return df
        missing = sorted(_SM_COLUMNS - set(df.columns))
        if missing:
            raise PointInTimeError(f"security master missing required columns: {missing}")
        temporal = ("valid_from", "available_time", "ingested_time")
        if any(df.schema[name] not in (pl.Date, pl.Datetime) for name in temporal):
            raise PointInTimeError("security master PIT timestamps must be Date or Datetime")
        valid_to_dtype = df.schema["valid_to"]
        if valid_to_dtype not in (pl.Date, pl.Datetime, pl.Null):
            raise PointInTimeError("security master valid_to must be Date, Datetime, or null")
        invalid = df.filter(
            pl.any_horizontal(
                pl.col("valid_from").is_null(),
                pl.col("available_time").is_null(),
                pl.col("ingested_time").is_null(),
                # Knowledge may arrive after valid_from (restatement); it cannot
                # be ingested before it was available.
                pl.col("available_time") > pl.col("ingested_time"),
                pl.col("security_id").is_null()
                | (pl.col("security_id").cast(pl.String).str.strip_chars() == ""),
                pl.col("ticker").is_null()
                | (pl.col("ticker").cast(pl.String).str.strip_chars() == ""),
                pl.col("source").is_null()
                | (pl.col("source").cast(pl.String).str.strip_chars() == ""),
            )
        )
        if invalid.height:
            raise PointInTimeError("security master contains null, blank, or impossible PIT fields")
        inverted = df.filter(
            pl.col("valid_to").is_not_null() & (pl.col("valid_to") < pl.col("valid_from"))
        )
        if inverted.height:
            raise PointInTimeError("security master contains valid_to earlier than valid_from")
        if df.select(["security_id", "valid_from"]).is_duplicated().any():
            raise PointInTimeError("security master contains duplicate security_id/valid_from rows")
        if df.select(["ticker", "valid_from"]).is_duplicated().any():
            raise PointInTimeError("security master contains duplicate ticker/valid_from rows")
        return df
