"""Point-in-time feature hooks over dipcatcher market-data adapters.

Bars come from anything with ``get_bars`` (:class:`quant_fund.data.protocols.MarketDataProvider`).
This module does not register a new vendor source. Resampling is a harness-side
OHLCV aggregation of those bars: ``src/quant_fund/data`` exposes adapters,
calendars, and PIT filters, and has no resample function of its own. Keeping
the aggregation here avoids coupling to a dataset-specific adapter.

Features at decision time ``t`` are computed only from bars with
``event_time <= t`` and ``available_time <= t``. The forecast target
``close[t+h] / close[t] - 1`` is not built here.
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from fx1.forecast.schema import validate_feature_schema
from quant_fund.data.point_in_time import PIT_COLS, filter_available, validate_feature_frame
from quant_fund.schemas.errors import PointInTimeError


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def normalize_bar_times(bars: pl.DataFrame) -> pl.DataFrame:
    """Cast bar timestamps to UTC datetimes so PIT comparisons are well-defined."""
    frame = bars
    for name in ("event_time", "available_time", "ingested_time"):
        if name not in frame.columns:
            continue
        dtype = frame.schema[name]
        if dtype == pl.Date:
            frame = frame.with_columns(
                pl.col(name).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
            )
        elif isinstance(dtype, pl.Datetime):
            if dtype.time_zone is None:
                frame = frame.with_columns(pl.col(name).dt.replace_time_zone("UTC"))
            else:
                frame = frame.with_columns(pl.col(name).dt.convert_time_zone("UTC"))
        else:
            raise PointInTimeError(f"{name} must be a Date or Datetime")
    return frame


def visible_bars(bars: pl.DataFrame, decision_time: datetime | None) -> pl.DataFrame:
    """Drop bars that were not observable at ``decision_time``.

    ``available_time`` is required. When the full dipcatcher PIT columns are
    present, :func:`quant_fund.data.point_in_time.filter_available` is used.
    """
    frame = normalize_bar_times(bars)
    if "available_time" not in frame.columns:
        raise PointInTimeError(
            "bars require available_time so a feature at t cannot use a later release"
        )
    if decision_time is None:
        return frame
    cutoff = as_utc(decision_time)
    if set(PIT_COLS).issubset(frame.columns):
        frame = filter_available(frame, cutoff)
    else:
        if frame.filter(pl.col("available_time").is_null()).height:
            raise PointInTimeError("null available_time")
        frame = frame.filter(pl.col("available_time") <= cutoff)
    frame = frame.filter(pl.col("event_time") <= cutoff)
    if frame.filter(pl.col("event_time") > pl.col("available_time")).height:
        raise PointInTimeError("event_time is after available_time")
    return frame


def resample_ohlcv(bars: pl.DataFrame, every: str) -> pl.DataFrame:
    """Aggregate OHLCV into non-overlapping buckets of width ``every``.

    The bucket timestamp is the last source ``event_time`` inside it, so the
    aggregate is not labeled before its last print. ``available_time`` is the
    max availability in the bucket and must be at or after that timestamp.
    """
    if not every or not str(every).strip():
        raise ValueError("resample every must be a non-empty polars duration, e.g. '1d'")
    frame = normalize_bar_times(bars)
    required = [
        "security_id",
        "event_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "available_time",
    ]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise PointInTimeError(f"resample input missing columns: {missing}")
    frame = frame.sort(["security_id", "event_time"])
    frame = frame.with_columns(pl.col("event_time").dt.truncate(every).alias("_bucket"))
    aggs: list[pl.Expr] = [
        pl.col("event_time").max().alias("event_time"),
        pl.col("open").sort_by("event_time").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").sort_by("event_time").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
        pl.col("available_time").max().alias("available_time"),
    ]
    if "ingested_time" in frame.columns:
        aggs.append(pl.col("ingested_time").max().alias("ingested_time"))
    if "source" in frame.columns:
        aggs.append(pl.col("source").n_unique().alias("_n_source"))
        aggs.append(pl.col("source").first().alias("source"))
    if "revision_id" in frame.columns:
        aggs.append(pl.col("revision_id").first().alias("revision_id"))
    out = frame.group_by(["security_id", "_bucket"], maintain_order=True).agg(aggs)
    if "_n_source" in out.columns:
        mixed = out.filter(pl.col("_n_source") > 1)
        if mixed.height:
            raise PointInTimeError("resample bucket mixes sources; refusing to collapse provenance")
        out = out.drop("_n_source")
    late = out.filter(pl.col("available_time") < pl.col("event_time"))
    if late.height:
        raise PointInTimeError("resampled available_time is before the bucket's last print")
    return out.drop("_bucket").sort(["security_id", "event_time"])


class OhlcvFeaturePipeline:
    """Trailing return, momentum, and volatility from the decision close.

    ``ret_1`` is ``close[t] / close[t-1] - 1`` (known at t). ``mom_k`` is the
    same over ``k`` bars. ``vol_w`` is the trailing standard deviation of
    ``ret_1``. None of these expressions shift forward.
    """

    def __init__(self, lookbacks: list[int], vol_window: int) -> None:
        windows = [int(k) for k in lookbacks]
        if not windows or any(k < 1 for k in windows):
            raise ValueError("lookbacks must be positive integers")
        if len(set(windows)) != len(windows):
            raise ValueError("lookbacks must be unique")
        if int(vol_window) < 2:
            raise ValueError("vol_window must be >= 2")
        self.lookbacks = tuple(sorted(windows))
        self.vol_window = int(vol_window)

    def feature_columns(self) -> list[str]:
        columns = ["ret_1"]
        columns.extend(f"mom_{k}" for k in self.lookbacks if k != 1)
        columns.append(f"vol_{self.vol_window}")
        return columns

    def build(
        self,
        bars: pl.DataFrame,
        *,
        decision_time: datetime | None = None,
    ) -> pl.DataFrame:
        frame = visible_bars(bars, decision_time)
        for name in ("security_id", "event_time", "close"):
            if name not in frame.columns:
                raise PointInTimeError(f"feature bars missing {name}")
        if frame.select(["security_id", "event_time"]).is_duplicated().any():
            raise PointInTimeError("duplicate (security_id, event_time) bars")
        frame = frame.sort(["security_id", "event_time"])
        frame = frame.with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("security_id") - 1.0).alias("ret_1")
        )
        mom_exprs = [
            (pl.col("close") / pl.col("close").shift(k).over("security_id") - 1.0).alias(f"mom_{k}")
            for k in self.lookbacks
            if k != 1
        ]
        if mom_exprs:
            frame = frame.with_columns(mom_exprs)
        vol_name = f"vol_{self.vol_window}"
        frame = frame.with_columns(
            pl.col("ret_1").rolling_std(self.vol_window).over("security_id").alias(vol_name)
        )
        columns = self.feature_columns()
        keep = ["event_time", "security_id", "close", "available_time", *columns]
        out = frame.select(keep).drop_nulls(subset=columns)
        if decision_time is not None and not out.is_empty():
            stamped = out.with_columns(pl.lit(as_utc(decision_time)).alias("decision_time"))
            validate_feature_frame(stamped, as_utc(decision_time))
        if out.is_empty():
            return out
        validate_feature_schema(out, columns)
        return out
