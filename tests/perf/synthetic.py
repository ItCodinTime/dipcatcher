"""Fixed synthetic panels for the performance suite.

Every series is a deterministic function of ``SEED``. These frames are
correctness and throughput inputs only — SYNTHETIC, not market evidence.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

SEED = 20260927
# Regular-session minute starts: 09:30 through 15:59 America/New_York.
RTH_MINUTES = 390


def _weekdays(n_days: int, *, start: date = date(2018, 1, 2)) -> pl.Series:
    days = pl.date_range(start, start + timedelta(days=n_days * 3 + 14), interval="1d", eager=True)
    return days.filter(days.dt.weekday() < 6).head(n_days)


def daily_ohlcv(n_symbols: int, n_days: int, *, seed: int = SEED) -> pl.DataFrame:
    """One row per symbol per weekday, with the columns feature builds require."""
    rng = np.random.default_rng(seed)
    days = _weekdays(n_days)
    times = days.cast(pl.Datetime("us")).dt.replace_time_zone("UTC").dt.offset_by("21h")
    sids = [f"S{i:04d}" for i in range(n_symbols)]
    shocks = rng.normal(0.0002, 0.012, size=(n_symbols, n_days))
    close = 50.0 * np.exp(np.cumsum(shocks, axis=1))
    open_ = close * (1.0 + rng.normal(0.0, 0.002, size=close.shape))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.003, size=close.shape)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.003, size=close.shape)))
    volume = rng.integers(100_000, 2_000_000, size=close.shape).astype(np.float64)
    n = n_symbols * n_days
    frame = pl.DataFrame(
        {
            "security_id": np.repeat(sids, n_days),
            "_t": np.tile(np.arange(n_days, dtype=np.int64), n_symbols),
            "close_total_return": close.reshape(n),
            "close": close.reshape(n),
            "open": open_.reshape(n),
            "volume": volume.reshape(n),
            "open_split_adjusted": open_.reshape(n),
            "close_split_adjusted": close.reshape(n),
            "high_split_adjusted": high.reshape(n),
            "low_split_adjusted": low.reshape(n),
            "source": np.repeat("synthetic", n),
        }
    ).join(pl.DataFrame({"_t": np.arange(n_days, dtype=np.int64), "event_time": times}), on="_t")
    return (
        frame.with_columns(pl.col("event_time").alias("available_time"))
        .drop("_t")
        .sort(["security_id", "event_time"])
    )


def minute_clock(n_sessions: int, *, start: date = date(2023, 1, 3)) -> pl.Series:
    """RTH minute-close timestamps (UTC) for ``n_sessions`` weekdays."""
    days = _weekdays(n_sessions, start=start)
    session_open = (
        days.cast(pl.Datetime("us")).dt.replace_time_zone("America/New_York").dt.offset_by("9h30m")
    )
    offsets = pl.Series("off", np.arange(RTH_MINUTES, dtype=np.int64))
    opens = pl.DataFrame({"session_open": session_open}).join(
        pl.DataFrame({"off": offsets}), how="cross"
    )
    return opens.select(
        (pl.col("session_open") + pl.duration(minutes=pl.col("off") + 1))
        .dt.convert_time_zone("UTC")
        .alias("event_time")
    )["event_time"]


def minute_bars(n_symbols: int, n_sessions: int, *, seed: int = SEED) -> pl.DataFrame:
    """Canonical 1-minute bars. OHLC stays inside the envelope."""
    from quant_fund.data.adapters.hf_ohlcv_1m import REVISION_ID, SOURCE_NAME

    rng = np.random.default_rng(seed + 1)
    event_times = minute_clock(n_sessions)
    n_bars = event_times.len()
    sids = [f"S{i:04d}" for i in range(n_symbols)]
    shocks = rng.normal(0.0, 0.0004, size=(n_symbols, n_bars))
    close = (40.0 * np.exp(np.cumsum(shocks, axis=1))).astype(np.float64)
    open_ = np.empty_like(close)
    open_[:, 0] = close[:, 0]
    open_[:, 1:] = close[:, :-1]
    wick = np.abs(rng.normal(0.0, 0.0004, size=close.shape))
    high = np.maximum(open_, close) * (1.0 + wick)
    low = np.minimum(open_, close) * (1.0 - wick)
    volume = rng.integers(100, 5_000, size=close.shape).astype(np.float64)
    n = n_symbols * n_bars
    frame = pl.DataFrame(
        {
            "security_id": np.repeat(sids, n_bars),
            "symbol": np.repeat(sids, n_bars),
            "_t": np.tile(np.arange(n_bars, dtype=np.int64), n_symbols),
            "open": open_.reshape(n),
            "high": high.reshape(n),
            "low": low.reshape(n),
            "close": close.reshape(n),
            "volume": volume.reshape(n),
        }
    ).join(
        pl.DataFrame({"_t": np.arange(n_bars, dtype=np.int64), "event_time": event_times}), on="_t"
    )
    ingested = datetime(2026, 1, 1, tzinfo=UTC)
    return frame.with_columns(
        pl.col("event_time").alias("available_time"),
        pl.lit(ingested).alias("ingested_time"),
        pl.lit(SOURCE_NAME).alias("source"),
        pl.lit(REVISION_ID).alias("revision_id"),
        pl.lit("USD").alias("currency"),
        pl.lit("rth").alias("session"),
    ).drop("_t")


def write_vendor_month(
    cache: Path,
    n_symbols: int,
    n_sessions: int,
    *,
    revision: str = "bench",
    seed: int = SEED,
) -> tuple[Path, list[str], date, date]:
    """Vendor month parquets under ``cache/revision``, plus the session bounds.

    One file per UTC month of the minute-open timestamp, which is how
    ``read_ohlcv_1m`` addresses the cache. The returned dates are the first
    and last America/New_York session dates in the panel.
    """
    bars = minute_bars(n_symbols, n_sessions, seed=seed)
    vendor = bars.select(
        (pl.col("event_time") - pl.duration(minutes=1)).alias("timestamp"),
        "open",
        "high",
        "low",
        "close",
        "volume",
        pl.col("symbol").alias("ticker"),
    ).with_columns(
        pl.col("timestamp").dt.year().alias("_year"),
        pl.col("timestamp").dt.month().alias("_month"),
    )
    dest = cache / revision
    dest.mkdir(parents=True, exist_ok=True)
    for year, month in vendor.select("_year", "_month").unique().iter_rows():
        part = vendor.filter((pl.col("_year") == year) & (pl.col("_month") == month)).drop(
            "_year", "_month"
        )
        part.write_parquet(dest / f"ohlcv_{int(year):04d}-{int(month):02d}.parquet")
    sessions = bars.select(
        (pl.col("event_time") - pl.duration(minutes=1))
        .dt.convert_time_zone("America/New_York")
        .dt.date()
        .alias("session_date")
    )["session_date"]
    symbols = [f"S{i:04d}" for i in range(n_symbols)]
    first = sessions.min()
    last = sessions.max()
    if not isinstance(first, date) or not isinstance(last, date):
        raise RuntimeError("synthetic minute panel has no session date")
    return cache, symbols, first, last


def dip_closes(n_symbols: int, n_days: int, *, seed: int = SEED) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed + 2)
    shocks = rng.normal(0.0003, 0.015, size=(n_symbols, n_days))
    closes = (100.0 * np.exp(np.cumsum(shocks, axis=1))).astype(np.float64)
    dates = [f"2020-01-{(i % 28) + 1:02d}" for i in range(n_days)]
    return closes, dates


def write_dip_dir(root: Path, n_symbols: int, n_days: int, *, seed: int = SEED) -> Path:
    closes, _dates = dip_closes(n_symbols, n_days, seed=seed)
    times = pl.datetime_range(
        datetime(2016, 1, 4),
        datetime(2016, 1, 4) + timedelta(days=n_days * 2),
        interval="1d",
        time_zone="UTC",
        eager=True,
    ).head(n_days)
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n_symbols):
        pl.DataFrame(
            {
                "symbol": [f"S{i:04d}"] * n_days,
                "event_time": times,
                "close": closes[i],
            }
        ).write_parquet(root / f"S{i:04d}_1d.parquet")
    return root
