"""Hugging Face ``mito0o852/OHLCV-1m`` as a canonical bar provider.

The Hub dataset is US-listed 1-minute OHLCV in monthly Parquet files
(``data/ohlcv_YYYY-MM.parquet``), timestamped at the **start** of each minute
in UTC. This adapter caches those months locally, maps them onto the bronze
bar contract, and refuses to repair bad prints.

Research plumbing only. The dataset card declares no license; upstream is
Finnhub. Prices are stored as received — adjustment is **not** declared.
``available_time`` is the minute-close convention, not a SIP vintage.
There is no broker path here.

Clock resampling lives in :func:`resample_ohlcv`. The repo has no other
clock-time OHLCV aggregator. Northset ``session_candles_from_daily``
fabricates pseudo-intraday paths and is intentionally not used.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import polars as pl

from quant_fund.data.calendars import session_days
from quant_fund.data.sources.base import HttpClient, SourceAdapter, SourceError, utc_now

DATASET_ID = "mito0o852/OHLCV-1m"
# Hub commit inspected 2026-09-26. resolve/ URLs pin this snapshot.
DATASET_REVISION = "776328445b7ac6e7815ef3a483e9c8ded1eb6d56"
SOURCE_NAME = "hf_ohlcv_1m"
# Adjustment is not declared on the dataset card. Do not read this as
# "vendor-adjusted" or "raw splits removed".
REVISION_ID = "HF_OHLCV_1M_UNDECLARED_ADJ"
ET = ZoneInfo("America/New_York")
USER_AGENT = "dipcatcher-research/1.0 (hf ohlcv-1m cache; not a live trading system)"

VENDOR_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume", "ticker")
# Regular session minute starts: 09:30 inclusive through 15:59 inclusive.
RTH_OPEN_MOD = 9 * 60 + 30
RTH_CLOSE_MOD = 16 * 60  # exclusive; last RTH start is 15:59
RTH_MINUTES = RTH_CLOSE_MOD - RTH_OPEN_MOD  # 390
EXT_OPEN_MOD = 4 * 60
EXT_CLOSE_MOD = 20 * 60
DEFAULT_MAX_MONTHS = 6
MAX_MONTH_BYTES = 1_000_000_000
_REVISION_OK = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")

Fetcher = Callable[[str, Path], None]
Clock = Callable[[], datetime]

_INTERVALS = {
    "1m": "1m",
    "1min": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "60m": "1h",
    "1h": "1h",
    "1d": "1d",
}

_Schema = dict[str, pl.DataType | type[pl.DataType]]

_BAR_SCHEMA: _Schema = {
    "security_id": pl.String,
    "symbol": pl.String,
    "event_time": pl.Datetime("us", "UTC"),
    "available_time": pl.Datetime("us", "UTC"),
    "ingested_time": pl.Datetime("us", "UTC"),
    "source": pl.String,
    "revision_id": pl.String,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "currency": pl.String,
    "session": pl.String,
}

_QUALITY_SCHEMA: _Schema = {
    "security_id": pl.String,
    "session_date": pl.Date,
    "n_rth": pl.Int64,
    "n_rth_expected": pl.Int64,
    "n_rth_missing": pl.Int64,
    "max_rth_gap_minutes": pl.Int64,
    "n_ext": pl.Int64,
    "n_off": pl.Int64,
    "n_missing_weekdays": pl.Int64,
}

_ACTION_SCHEMA: _Schema = {
    "security_id": pl.String,
    "event_time": pl.Datetime("us", "UTC"),
    "available_time": pl.Datetime("us", "UTC"),
    "ingested_time": pl.Datetime("us", "UTC"),
    "source": pl.String,
    "revision_id": pl.String,
    "action_type": pl.String,
    "factor": pl.Float64,
    "amount": pl.Float64,
    "new_ticker": pl.String,
}

_MASTER_SCHEMA: _Schema = {
    "security_id": pl.String,
    "ticker": pl.String,
    "name": pl.String,
    "exchange": pl.String,
    "currency": pl.String,
    "sector": pl.String,
    "industry": pl.String,
    "security_type": pl.String,
    "valid_from": pl.Datetime("us", "UTC"),
    "valid_to": pl.Datetime("us", "UTC"),
    "available_time": pl.Datetime("us", "UTC"),
    "ingested_time": pl.Datetime("us", "UTC"),
    "source": pl.String,
    "revision_id": pl.String,
}


class OhlcvQualityError(SourceError):
    """A requested slice failed a data-quality check. Nothing was rewritten."""


class MonthNotFound(SourceError):
    """The monthly Parquet object is not on the Hub (HTTP 404)."""


@dataclass(frozen=True)
class OhlcvSlice:
    """Resampled bars, the 1-minute slice they came from, and the gap report."""

    bars: pl.DataFrame
    minutes: pl.DataFrame
    quality: pl.DataFrame


def default_cache_dir() -> Path:
    """Local month cache. Override with ``HF_OHLCV_1M_CACHE``."""
    return Path(os.environ.get("HF_OHLCV_1M_CACHE", "data/hf_ohlcv_1m"))


def month_parquet_url(year: int, month: int, revision: str = DATASET_REVISION) -> str:
    """HTTPS URL of one monthly file at a pinned dataset revision."""
    _require_revision(revision)
    if not 1 <= int(month) <= 12:
        raise ValueError("month must be in 1..12")
    return (
        "https://huggingface.co/datasets/mito0o852/OHLCV-1m/resolve/"
        f"{revision}/data/ohlcv_{int(year):04d}-{int(month):02d}.parquet"
    )


def parse_symbols(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """Split a comma-separated ticker list. Matching is uppercase, dots kept."""
    if value is None:
        return []
    parts = value.split(",") if isinstance(value, str) else list(value)
    out: list[str] = []
    for part in parts:
        symbol = str(part).strip().upper()
        if not symbol:
            continue
        if any(ch in symbol for ch in "/\\") or any(ch.isspace() for ch in symbol):
            raise ValueError(f"unsafe ticker {part!r}")
        out.append(symbol)
    return list(dict.fromkeys(out))


def normalize_interval(value: str) -> str:
    """Map a bar size onto ``1m``, ``5m``, ``15m``, ``30m``, ``1h``, or ``1d``.

    ``1M`` is rejected so it cannot collapse into ``1m`` the way a case-fold
    would. Monthly bars are not produced here.
    """
    text = str(value).strip()
    if text == "1M":
        raise ValueError("monthly resample is not supported; use 1m, 5m, 15m, 30m, 1h, or 1d")
    key = _INTERVALS.get(text.lower())
    if key is None:
        raise ValueError(f"unsupported interval {value!r}; expected 1m, 5m, 15m, 30m, 1h, or 1d")
    return key


def parse_bound(value: datetime | date | str, *, role: str) -> datetime:
    """Inclusive UTC bound on canonical ``event_time`` (minute close).

    A calendar ``date`` or ``YYYY-MM-DD`` string is the America/New_York
    session day: start is 00:00 ET, end is one microsecond before the next
    midnight ET. Aware datetimes are converted to UTC. Naive datetimes fail.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise OhlcvQualityError(f"{role} datetime must be timezone-aware")
        return value.astimezone(UTC)
    if isinstance(value, date):
        return _et_day_bound(value, role=role)
    text = str(value).strip()
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        try:
            day = date.fromisoformat(text)
        except ValueError as exc:
            raise OhlcvQualityError(f"{role} is not a calendar date: {text}") from exc
        return _et_day_bound(day, role=role)
    parsed = _parse_aware(text, role=role)
    return parsed.astimezone(UTC)


def empty_bars(*, resampled: bool = False) -> pl.DataFrame:
    schema = dict(_BAR_SCHEMA)
    if resampled:
        schema["n_source_minutes"] = pl.Int64
    return pl.DataFrame(schema=schema)


def empty_quality() -> pl.DataFrame:
    return pl.DataFrame(schema=_QUALITY_SCHEMA)


def empty_corporate_actions() -> pl.DataFrame:
    """The dataset ships no corporate-action table. This frame stays empty."""
    return pl.DataFrame(schema=_ACTION_SCHEMA)


def normalize_vendor_frame(frame: pl.DataFrame, *, clock: Clock | None = None) -> pl.DataFrame:
    """Map vendor rows onto the bronze bar schema. Invalid rows raise.

    ``timestamp`` is the minute open. ``event_time`` and ``available_time``
    are that instant plus one minute (the close). Session labels use the
    minute **open** in America/New_York:

    - ``rth``: weekdays, 09:30 inclusive through 16:00 exclusive
    - ``ext``: weekdays, 04:00–09:30 or 16:00–20:00
    - ``off``: anything else (kept, not relabeled)
    """
    missing = [name for name in VENDOR_COLUMNS if name not in frame.columns]
    if missing:
        raise OhlcvQualityError(f"vendor frame missing columns: {missing}")
    if frame.is_empty():
        return empty_bars()
    stamped = _prepare_vendor_clock(frame)
    _reject_bad_vendor_rows(stamped)
    now = _require_clock(clock or utc_now)
    out = stamped.select(
        pl.col("ticker").alias("security_id"),
        pl.col("ticker").alias("symbol"),
        (pl.col("timestamp") + pl.duration(minutes=1))
        .dt.convert_time_zone("UTC")
        .cast(pl.Datetime("us", "UTC"))
        .alias("event_time"),
        pl.col("open"),
        pl.col("high"),
        pl.col("low"),
        pl.col("close"),
        pl.col("volume"),
        pl.col("_session").alias("session"),
    ).with_columns(
        pl.col("event_time").alias("available_time"),
        pl.lit(now).cast(pl.Datetime("us", "UTC")).alias("ingested_time"),
        pl.lit(SOURCE_NAME).alias("source"),
        pl.lit(REVISION_ID).alias("revision_id"),
        pl.lit("USD").alias("currency"),
    )
    if out.filter(pl.col("ingested_time") < pl.col("available_time")).height:
        raise OhlcvQualityError("ingested_time is earlier than the minute close")
    if out.select(["security_id", "event_time"]).is_duplicated().any():
        raise OhlcvQualityError("duplicate security_id/event_time after timestamp normalization")
    columns = list(_BAR_SCHEMA)
    return out.select(columns).sort(["security_id", "event_time"])


def minute_gap_report(bars: pl.DataFrame) -> pl.DataFrame:
    """Flag missing regular-session minutes. Does not insert bars.

    Each observed America/New_York session date is compared with the 390
    minute starts from 09:30 through 15:59. Early closes and holidays inside
    a name's span show up as gaps: the weekday calendar has no holiday set,
    and this report does not invent one. ``n_missing_weekdays`` counts
    weekdays between a name's first and last observed session date that have
    no print at all.
    """
    if bars.is_empty():
        return empty_quality()
    required = ("security_id", "event_time", "session")
    missing = [name for name in required if name not in bars.columns]
    if missing:
        raise OhlcvQualityError(f"gap report missing columns: {missing}")
    work = bars.with_columns(
        (pl.col("event_time") - pl.duration(minutes=1))
        .dt.convert_time_zone("America/New_York")
        .alias("_start")
    ).with_columns(
        pl.col("_start").dt.date().alias("session_date"),
        (
            pl.col("_start").dt.hour().cast(pl.Int32) * 60
            + pl.col("_start").dt.minute().cast(pl.Int32)
        ).alias("_mod"),
    )
    base = work.group_by(["security_id", "session_date"]).agg(
        (pl.col("session") == "rth").sum().cast(pl.Int64).alias("n_rth"),
        (pl.col("session") == "ext").sum().cast(pl.Int64).alias("n_ext"),
        (pl.col("session") == "off").sum().cast(pl.Int64).alias("n_off"),
    )
    rth = work.filter(pl.col("session") == "rth").sort(["security_id", "session_date", "_mod"])
    if rth.is_empty():
        steps = pl.DataFrame(
            schema={
                "security_id": pl.String,
                "session_date": pl.Date,
                "first_mod": pl.Int32,
                "last_mod": pl.Int32,
                "max_step": pl.Int32,
            }
        )
    else:
        steps = (
            rth.with_columns(
                pl.col("_mod").diff().over(["security_id", "session_date"]).alias("_step")
            )
            .group_by(["security_id", "session_date"])
            .agg(
                pl.col("_mod").min().alias("first_mod"),
                pl.col("_mod").max().alias("last_mod"),
                pl.col("_step").max().alias("max_step"),
            )
        )
    joined = base.join(steps, on=["security_id", "session_date"], how="left")
    internal = (pl.col("max_step").fill_null(1) - 1).clip(lower_bound=0)
    open_gap = (pl.col("first_mod") - RTH_OPEN_MOD).fill_null(RTH_MINUTES).clip(lower_bound=0)
    close_gap = (RTH_CLOSE_MOD - 1 - pl.col("last_mod")).fill_null(RTH_MINUTES).clip(lower_bound=0)
    joined = joined.with_columns(
        pl.lit(RTH_MINUTES).alias("n_rth_expected"),
        (pl.lit(RTH_MINUTES) - pl.col("n_rth")).alias("n_rth_missing"),
        pl.max_horizontal(internal, open_gap, close_gap)
        .cast(pl.Int64)
        .alias("max_rth_gap_minutes"),
    )
    weekday_gaps = _missing_weekdays(joined)
    gap_frame = pl.DataFrame(
        {
            "security_id": list(weekday_gaps),
            "n_missing_weekdays": [weekday_gaps[key] for key in weekday_gaps],
        }
    )
    if gap_frame.is_empty():
        joined = joined.with_columns(pl.lit(0).cast(pl.Int64).alias("n_missing_weekdays"))
    else:
        joined = joined.join(gap_frame, on="security_id", how="left")
    return joined.select(list(_QUALITY_SCHEMA)).sort(["security_id", "session_date"])


def resample_ohlcv(
    bars: pl.DataFrame,
    every: str,
    *,
    session: str | None = None,
) -> pl.DataFrame:
    """Aggregate canonical 1-minute bars. Empty clock bins are not filled.

    Intraday bins align to America/New_York. Five-, fifteen-, and
    thirty-minute bins share the 09:30 open. Hourly bins are offset 30
    minutes so they run 09:30–10:30, not 09:00–10:00. Daily bars group by
    the New York session date, not UTC midnight.

    ``event_time`` stays the close of the last **observed** minute in the
    bin. ``n_source_minutes`` is that count, so a gappy 5-minute bin stays
    visibly short of five. Bins that mix ``rth`` and ``ext`` are labeled
    ``mixed``.
    """
    interval = normalize_interval(every)
    frame = bars
    if session is not None:
        if session not in {"rth", "ext", "off"}:
            raise ValueError("session filter must be rth, ext, or off")
        if "session" not in frame.columns:
            raise OhlcvQualityError("resample session filter requires a session column")
        frame = frame.filter(pl.col("session") == session)
    if interval == "1m":
        return frame.sort(["security_id", "event_time"]) if not frame.is_empty() else frame
    if frame.is_empty():
        return empty_bars(resampled=True)
    if interval == "1d":
        out = _resample_session_days(frame)
    else:
        out = _resample_intraday(frame, interval)
    _reject_resampled_envelope(out)
    return out.sort(["security_id", "event_time"])


def read_ohlcv_1m(
    *,
    symbols: str | list[str] | tuple[str, ...] | None,
    cache_dir: str | Path | None = None,
    start: datetime | date | str | None = None,
    end: datetime | date | str | None = None,
    revision: str = DATASET_REVISION,
    allow_download: bool = False,
    interval: str = "1m",
    session: str | None = None,
    max_months: int = DEFAULT_MAX_MONTHS,
    strict_gaps: bool = False,
    strict_off: bool = False,
    fetcher: Fetcher | None = None,
    clock: Clock | None = None,
) -> OhlcvSlice:
    """Load one symbol/date slice from the local month cache or the Hub.

    Downloads happen only when ``allow_download`` is true. A missing cached
    month otherwise raises. The default span cap is six calendar months of
    files so a call cannot pull the ~88 GB corpus by accident.
    """
    tickers = parse_symbols(symbols)
    if not tickers:
        raise ValueError("symbols are required (comma-separated US tickers)")
    if max_months < 1:
        raise ValueError("max_months must be >= 1")
    interval_key = normalize_interval(interval)
    if session is not None and session not in {"rth", "ext", "off"}:
        raise ValueError("session filter must be rth, ext, or off")
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    _require_revision(revision)
    start_utc = None if start is None else parse_bound(start, role="start")
    end_utc = None if end is None else parse_bound(end, role="end")
    if start_utc is not None and end_utc is not None and start_utc > end_utc:
        raise OhlcvQualityError("start is after end")
    paths = _month_paths(
        cache=cache,
        revision=revision,
        start=start_utc,
        end=end_utc,
        max_months=max_months,
        allow_download=allow_download,
        fetcher=fetcher or http_download,
    )
    vendor = _scan_vendor(paths, tickers, start=start_utc, end=end_utc)
    found = set(vendor["ticker"].unique().to_list()) if not vendor.is_empty() else set()
    absent = [symbol for symbol in tickers if symbol not in found]
    if absent:
        raise OhlcvQualityError(
            "symbols have no rows in the requested months "
            f"{[path.name for path in paths]}: {absent}"
        )
    minutes = normalize_vendor_frame(vendor, clock=clock)
    quality = minute_gap_report(minutes)
    _enforce_quality(quality, strict_gaps=strict_gaps, strict_off=strict_off)
    if session is not None or interval_key != "1m":
        bars = resample_ohlcv(minutes, interval_key, session=session)
    else:
        bars = minutes
    return OhlcvSlice(bars=bars, minutes=minutes, quality=quality)


def security_master_from_bars(bars: pl.DataFrame, *, clock: Clock | None = None) -> pl.DataFrame:
    """Identity rows for the slice. Exchange is UNKNOWN: the file has none.

    ``security_type`` is ``unknown`` because the tape mixes common stock,
    preferreds (dotted tickers), warrants, and funds. No listing window is
    invented beyond the first observed print; ``valid_to`` stays null.
    """
    if bars.is_empty():
        return pl.DataFrame(schema=_MASTER_SCHEMA)
    now = _require_clock(clock or utc_now)
    rows: list[dict[str, object]] = []
    for security_id in bars["security_id"].unique().sort().to_list():
        name_bars = bars.filter(pl.col("security_id") == security_id)
        first = name_bars["event_time"].min()
        if not isinstance(first, datetime):
            raise OhlcvQualityError(f"security {security_id} has no event_time")
        rows.append(
            {
                "security_id": str(security_id),
                "ticker": str(security_id),
                "name": str(security_id),
                "exchange": "UNKNOWN",
                "currency": "USD",
                "sector": "Unknown",
                "industry": "Unknown",
                "security_type": "unknown",
                "valid_from": first,
                "valid_to": None,
                "available_time": first,
                "ingested_time": now,
                "source": SOURCE_NAME,
                "revision_id": REVISION_ID,
            }
        )
    return pl.DataFrame(rows, schema=_MASTER_SCHEMA).sort("security_id")


def http_download(url: str, dest: Path) -> None:
    """Stream one HTTPS object to ``dest``. Caller checks the Parquet footer."""
    stream_https(url, dest, opener=urlopen, max_bytes=MAX_MONTH_BYTES, timeout=180.0)


def stream_https(
    url: str,
    dest: Path,
    *,
    opener: Callable[..., Any],
    max_bytes: int,
    timeout: float,
) -> None:
    """Download ``url`` with a hard byte cap. Oversize and HTTP errors raise."""
    if urlsplit(url).scheme.lower() != "https":
        raise SourceError(f"refusing non-HTTPS dataset URL: {url}")
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    request = Request(url, headers={"User-Agent": USER_AGENT}, method="GET")
    try:
        response: Any = opener(request, timeout=timeout)  # noqa: S310  # nosec B310
    except HTTPError as exc:
        dest.unlink(missing_ok=True)
        if exc.code == 404:
            raise MonthNotFound(f"monthly parquet not found: {url}") from exc
        raise SourceError(f"GET failed ({exc.code}): {url}") from exc
    except (OSError, URLError, TimeoutError) as exc:
        dest.unlink(missing_ok=True)
        raise SourceError(f"GET failed: {url}") from exc
    total = 0
    try:
        with response, dest.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise SourceError(
                        f"monthly parquet exceeded {max_bytes} bytes; download aborted: {url}"
                    )
                handle.write(chunk)
    except SourceError:
        dest.unlink(missing_ok=True)
        raise
    except (OSError, URLError, TimeoutError) as exc:
        dest.unlink(missing_ok=True)
        raise SourceError(f"GET failed while streaming: {url}") from exc
    if total < 1:
        dest.unlink(missing_ok=True)
        raise SourceError(f"empty response: {url}")


class HfOhlcv1mProvider:
    """``MarketDataProvider`` over a local month cache.

    ``allow_download`` defaults off so ingest and research runs stay offline.
    ``dipcatcher collect --source hf_ohlcv_1m`` is the network entry.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        symbols: str | list[str] | tuple[str, ...] | None = None,
        start: datetime | date | str | None = None,
        end: datetime | date | str | None = None,
        revision: str = DATASET_REVISION,
        interval: str = "1m",
        session: str | None = None,
        allow_download: bool = False,
        max_months: int = DEFAULT_MAX_MONTHS,
        strict_gaps: bool = False,
        strict_off: bool = False,
        fetcher: Fetcher | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        self.symbols = parse_symbols(symbols)
        self.start = start
        self.end = end
        self.revision = revision
        self.interval = normalize_interval(interval)
        self.session = session
        self.allow_download = bool(allow_download)
        self.max_months = int(max_months)
        self.strict_gaps = bool(strict_gaps)
        self.strict_off = bool(strict_off)
        self.fetcher = fetcher or http_download
        self.clock = clock or utc_now
        self.quality_report: pl.DataFrame = empty_quality()
        self._last_minute_bars: pl.DataFrame | None = None

    def get_bars(
        self,
        start: datetime | date | str | None = None,
        end: datetime | date | str | None = None,
        security_ids: list[str] | None = None,
        *,
        session: str | None = None,
        interval: str | None = None,
        strict_gaps: bool | None = None,
        strict_off: bool | None = None,
    ) -> pl.DataFrame:
        tickers = parse_symbols(security_ids) if security_ids is not None else self.symbols
        result = read_ohlcv_1m(
            symbols=tickers,
            cache_dir=self.cache_dir,
            start=self.start if start is None else start,
            end=self.end if end is None else end,
            revision=self.revision,
            allow_download=self.allow_download,
            interval=self.interval if interval is None else interval,
            session=self.session if session is None else session,
            max_months=self.max_months,
            strict_gaps=self.strict_gaps if strict_gaps is None else strict_gaps,
            strict_off=self.strict_off if strict_off is None else strict_off,
            fetcher=self.fetcher,
            clock=self.clock,
        )
        self.quality_report = result.quality
        self._last_minute_bars = result.minutes
        return result.bars

    def get_corporate_actions(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        del start, end
        return empty_corporate_actions()

    def get_security_master(self) -> pl.DataFrame:
        bars = self._last_minute_bars
        if bars is None:
            # Identity is taken from the 1-minute slice, even if get_bars
            # was asked for a coarser interval.
            loaded = read_ohlcv_1m(
                symbols=self.symbols,
                cache_dir=self.cache_dir,
                start=self.start,
                end=self.end,
                revision=self.revision,
                allow_download=self.allow_download,
                interval="1m",
                session=None,
                max_months=self.max_months,
                strict_gaps=False,
                strict_off=False,
                fetcher=self.fetcher,
                clock=self.clock,
            )
            bars = loaded.minutes
            self._last_minute_bars = bars
            if self.quality_report.is_empty():
                self.quality_report = loaded.quality
        return security_master_from_bars(bars, clock=self.clock)


class HfOhlcv1mSource(SourceAdapter):
    """Registry adapter. ``fetch`` may download; ingest does not call it."""

    name = SOURCE_NAME

    def __init__(
        self,
        client: HttpClient | None = None,
        *,
        fetcher: Fetcher | None = None,
        clock: Clock | None = None,
    ) -> None:
        super().__init__(client)
        self.fetcher = fetcher or http_download
        self.clock = clock or utc_now
        self.quality_report: pl.DataFrame = empty_quality()

    def fetch(self, **kwargs: object) -> pl.DataFrame:
        allowed = {
            "symbols",
            "symbol",
            "start",
            "end",
            "cache_dir",
            "interval",
            "session",
            "max_months",
            "strict_gaps",
            "strict_off",
            "revision",
        }
        unknown = sorted(set(kwargs) - allowed)
        if unknown:
            raise ValueError(f"unknown hf_ohlcv_1m parameters: {unknown}")
        raw_symbols = kwargs.get("symbols", kwargs.get("symbol"))
        if not isinstance(raw_symbols, str | list | tuple) and raw_symbols is not None:
            raise ValueError("symbols must be a string or list of tickers")
        interval = kwargs.get("interval", "1m")
        revision = kwargs.get("revision", DATASET_REVISION)
        if not isinstance(interval, str) or not isinstance(revision, str):
            raise ValueError("interval and revision must be strings")
        cache = kwargs.get("cache_dir")
        if cache is not None and not isinstance(cache, str | Path):
            raise ValueError("cache_dir must be a path")
        provider = HfOhlcv1mProvider(
            cache_dir=cache if isinstance(cache, str | Path) else default_cache_dir(),
            symbols=raw_symbols if isinstance(raw_symbols, str | list | tuple) else None,
            start=_optional_bound(kwargs.get("start")),
            end=_optional_bound(kwargs.get("end")),
            revision=revision,
            interval=interval,
            session=_optional_session(kwargs.get("session")),
            allow_download=True,
            max_months=_optional_int(kwargs.get("max_months"), default=DEFAULT_MAX_MONTHS),
            strict_gaps=_optional_bool(kwargs.get("strict_gaps"), default=False),
            strict_off=_optional_bool(kwargs.get("strict_off"), default=False),
            fetcher=self.fetcher,
            clock=self.clock,
        )
        frame = provider.get_bars()
        self.quality_report = provider.quality_report
        return frame


def _optional_bound(value: object) -> datetime | date | str | None:
    if value is None:
        return None
    if isinstance(value, datetime | date | str):
        return value
    raise ValueError("start and end must be dates, datetimes, or strings")


def _optional_session(value: object) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError("session must be a string")
    return value


def _optional_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes"}:
        return True
    if isinstance(value, str) and value.strip().lower() in {"0", "false", "no"}:
        return False
    raise ValueError(f"expected a boolean, got {value!r}")


def _optional_int(value: object, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise ValueError(f"expected an integer, got {value!r}")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"expected an integer, got {value!r}") from exc


def _require_revision(revision: str) -> None:
    if not revision or any(ch not in _REVISION_OK for ch in revision) or len(revision) > 80:
        raise ValueError(f"unsafe dataset revision {revision!r}")


def _et_day_bound(day: date, *, role: str) -> datetime:
    if role == "end":
        nxt = datetime(day.year, day.month, day.day, tzinfo=ET) + timedelta(days=1)
        return (nxt - timedelta(microseconds=1)).astimezone(UTC)
    start = datetime(day.year, day.month, day.day, tzinfo=ET)
    return start.astimezone(UTC)


def _parse_aware(text: str, *, role: str) -> datetime:
    raw = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise OhlcvQualityError(f"{role} is not an ISO timestamp: {text}") from exc
    if parsed.tzinfo is None:
        raise OhlcvQualityError(f"{role} datetime must be timezone-aware ({text})")
    return parsed


def _require_clock(clock: Clock) -> datetime:
    now = clock()
    if now.tzinfo is None:
        raise OhlcvQualityError("clock returned a naive datetime")
    return now.astimezone(UTC)


def _prepare_vendor_clock(frame: pl.DataFrame) -> pl.DataFrame:
    dtype = frame.schema["timestamp"]
    if not isinstance(dtype, pl.Datetime) or dtype.time_zone is None:
        raise OhlcvQualityError("timestamp must be a timezone-aware datetime")
    stamped = frame.with_columns(
        pl.col("ticker").cast(pl.String).str.strip_chars().str.to_uppercase().alias("ticker"),
        pl.col("timestamp").dt.convert_time_zone("UTC").alias("timestamp"),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
    )
    if stamped.filter(pl.col("ticker").str.contains(r"\s")).height:
        raise OhlcvQualityError("ticker contains whitespace; rows were not rewritten")
    start_et = pl.col("timestamp").dt.convert_time_zone("America/New_York")
    mod = start_et.dt.hour().cast(pl.Int32) * 60 + start_et.dt.minute().cast(pl.Int32)
    weekend = start_et.dt.weekday() >= 6
    rth = (~weekend) & (mod >= RTH_OPEN_MOD) & (mod < RTH_CLOSE_MOD)
    ext = (~weekend) & (
        ((mod >= EXT_OPEN_MOD) & (mod < RTH_OPEN_MOD))
        | ((mod >= RTH_CLOSE_MOD) & (mod < EXT_CLOSE_MOD))
    )
    return stamped.with_columns(
        pl.when(rth)
        .then(pl.lit("rth"))
        .when(ext)
        .then(pl.lit("ext"))
        .otherwise(pl.lit("off"))
        .alias("_session")
    )


def _reject_bad_vendor_rows(frame: pl.DataFrame) -> None:
    if frame.select(["ticker", "timestamp"]).is_duplicated().any():
        dup = (
            frame.filter(pl.struct(["ticker", "timestamp"]).is_duplicated())
            .select(["ticker", "timestamp"])
            .unique()
            .head(5)
        )
        raise OhlcvQualityError(f"duplicate ticker/timestamp rows (not dropped): {dup.to_dicts()}")
    misaligned = (
        (pl.col("timestamp").dt.second() != 0)
        | (pl.col("timestamp").dt.microsecond() != 0)
        | (pl.col("timestamp").dt.nanosecond() != 0)
    )
    reason = pl.concat_str(
        [
            pl.when(pl.col("timestamp").is_null())
            .then(pl.lit("null timestamp;"))
            .otherwise(pl.lit("")),
            pl.when(pl.col("ticker").is_null() | (pl.col("ticker") == ""))
            .then(pl.lit("blank ticker;"))
            .otherwise(pl.lit("")),
            pl.when(misaligned)
            .then(pl.lit("timestamp is not minute-aligned;"))
            .otherwise(pl.lit("")),
            *[
                pl.when(pl.col(name).is_null() | ~pl.col(name).is_finite() | (pl.col(name) <= 0))
                .then(pl.lit(f"non-positive {name};"))
                .otherwise(pl.lit(""))
                for name in ("open", "high", "low", "close")
            ],
            pl.when(
                pl.col("volume").is_null() | ~pl.col("volume").is_finite() | (pl.col("volume") < 0)
            )
            .then(pl.lit("negative or non-finite volume;"))
            .otherwise(pl.lit("")),
            pl.when(pl.col("high") < pl.col("low"))
            .then(pl.lit("high < low;"))
            .otherwise(pl.lit("")),
            pl.when(pl.col("open") > pl.col("high"))
            .then(pl.lit("open > high;"))
            .otherwise(pl.lit("")),
            pl.when(pl.col("open") < pl.col("low"))
            .then(pl.lit("open < low;"))
            .otherwise(pl.lit("")),
            pl.when(pl.col("close") > pl.col("high"))
            .then(pl.lit("close > high;"))
            .otherwise(pl.lit("")),
            pl.when(pl.col("close") < pl.col("low"))
            .then(pl.lit("close < low;"))
            .otherwise(pl.lit("")),
        ]
    ).alias("_reason")
    flagged = frame.with_columns(reason).filter(pl.col("_reason") != "")
    if flagged.is_empty():
        return
    sample = flagged.select(["ticker", "timestamp", "_reason"]).head(5).to_dicts()
    extra = flagged.height - len(sample)
    tail = f" (+{extra} more)" if extra > 0 else ""
    raise OhlcvQualityError(f"bad bars were not repaired: {sample}{tail}")


def _missing_weekdays(report: pl.DataFrame) -> dict[str, int]:
    gaps: dict[str, int] = {}
    if report.is_empty():
        return gaps
    for security_id, group in report.group_by("security_id"):
        key = security_id[0] if isinstance(security_id, tuple) else security_id
        days = [day for day in group["session_date"].to_list() if isinstance(day, date)]
        if not days:
            gaps[str(key)] = 0
            continue
        calendar = set(session_days(min(days), max(days)))
        observed = {day for day in days if day.weekday() < 5}
        gaps[str(key)] = len(calendar - observed)
    return gaps


def _enforce_quality(quality: pl.DataFrame, *, strict_gaps: bool, strict_off: bool) -> None:
    if quality.is_empty():
        return
    if strict_gaps:
        missing = int(quality["n_rth_missing"].sum())
        if missing > 0:
            sample = quality.filter(pl.col("n_rth_missing") > 0).head(3).to_dicts()
            raise OhlcvQualityError(
                f"strict_gaps: {missing} missing regular-session minutes (not filled): {sample}"
            )
    if strict_off:
        off = int(quality["n_off"].sum())
        if off > 0:
            sample = quality.filter(pl.col("n_off") > 0).head(3).to_dicts()
            raise OhlcvQualityError(f"strict_off: {off} off-session bars (not dropped): {sample}")


def _resample_session_days(frame: pl.DataFrame) -> pl.DataFrame:
    work = frame.with_columns(
        (pl.col("event_time") - pl.duration(minutes=1))
        .dt.convert_time_zone("America/New_York")
        .dt.date()
        .alias("_session_date")
    ).sort(["security_id", "event_time"])
    out = work.group_by(["security_id", "_session_date"], maintain_order=True).agg(
        pl.col("symbol").first(),
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
        pl.col("volume").sum(),
        pl.col("event_time").last(),
        pl.col("ingested_time").max(),
        pl.col("source").first(),
        pl.col("revision_id").first(),
        pl.col("currency").first(),
        pl.len().cast(pl.Int64).alias("n_source_minutes"),
        pl.col("session").unique().alias("_sessions"),
    )
    return _finish_resampled(out)


def _resample_intraday(frame: pl.DataFrame, interval: str) -> pl.DataFrame:
    every = {"5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h"}[interval]
    offset = "30m" if interval == "1h" else "0m"
    work = frame.with_columns(
        (pl.col("event_time") - pl.duration(minutes=1))
        .dt.convert_time_zone("America/New_York")
        .alias("_start_et")
    ).sort(["security_id", "_start_et"])
    out = work.group_by_dynamic(
        "_start_et",
        every=every,
        offset=offset,
        closed="left",
        label="left",
        group_by="security_id",
    ).agg(
        pl.col("symbol").first(),
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
        pl.col("volume").sum(),
        pl.col("event_time").last(),
        pl.col("ingested_time").max(),
        pl.col("source").first(),
        pl.col("revision_id").first(),
        pl.col("currency").first(),
        pl.len().cast(pl.Int64).alias("n_source_minutes"),
        pl.col("session").unique().alias("_sessions"),
    )
    return _finish_resampled(out.drop("_start_et"))


def _finish_resampled(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.with_columns(
            pl.col("event_time").dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC")),
            pl.when(pl.col("_sessions").list.len() == 1)
            .then(pl.col("_sessions").list.first())
            .otherwise(pl.lit("mixed"))
            .alias("session"),
        )
        .with_columns(pl.col("event_time").alias("available_time"))
        .select(
            [
                "security_id",
                "symbol",
                "event_time",
                "available_time",
                "ingested_time",
                "source",
                "revision_id",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "currency",
                "session",
                "n_source_minutes",
            ]
        )
    )


def _reject_resampled_envelope(frame: pl.DataFrame) -> None:
    if frame.is_empty():
        return
    bad = frame.filter(
        (pl.col("high") < pl.col("low"))
        | (pl.col("open") > pl.col("high"))
        | (pl.col("open") < pl.col("low"))
        | (pl.col("close") > pl.col("high"))
        | (pl.col("close") < pl.col("low"))
        | (pl.col("volume") < 0)
    )
    if bad.height:
        raise OhlcvQualityError("resampled bars broke the OHLC envelope; refusing the aggregate")


def _month_paths(
    *,
    cache: Path,
    revision: str,
    start: datetime | None,
    end: datetime | None,
    max_months: int,
    allow_download: bool,
    fetcher: Fetcher,
) -> list[Path]:
    if start is None and end is None:
        paths = _list_cached_months(cache, revision)
        if not paths:
            raise OhlcvQualityError(
                f"no cached months under {cache / revision}; "
                "run `dipcatcher collect --source hf_ohlcv_1m` or pass start/end with a cache"
            )
        if len(paths) > max_months:
            raise OhlcvQualityError(
                f"cached months ({len(paths)}) exceed max_months={max_months}; pass start and end"
            )
        return paths
    if start is None or end is None:
        raise OhlcvQualityError("pass both start and end, or neither to read the local cache")
    vendor_start = start - timedelta(minutes=1)
    vendor_end = end - timedelta(minutes=1)
    months = _months_between(vendor_start, vendor_end)
    if len(months) > max_months:
        raise OhlcvQualityError(
            f"requested span covers {len(months)} monthly files; max_months={max_months}"
        )
    return [
        _ensure_month(
            cache=cache,
            revision=revision,
            year=year,
            month=month,
            allow_download=allow_download,
            fetcher=fetcher,
        )
        for year, month in months
    ]


def _months_between(start: datetime, end: datetime) -> list[tuple[int, int]]:
    if start > end:
        raise OhlcvQualityError("start is after end")
    year, month = start.year, start.month
    out: list[tuple[int, int]] = []
    while (year, month) <= (end.year, end.month):
        out.append((year, month))
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
    return out


def _list_cached_months(cache: Path, revision: str) -> list[Path]:
    root = cache / revision
    if not root.is_dir():
        return []
    return sorted(path for path in root.glob("ohlcv_????-??.parquet") if path.is_file())


def _ensure_month(
    *,
    cache: Path,
    revision: str,
    year: int,
    month: int,
    allow_download: bool,
    fetcher: Fetcher,
) -> Path:
    path = cache / revision / f"ohlcv_{year:04d}-{month:02d}.parquet"
    if path.exists():
        _require_vendor_parquet(path)
        return path
    if not allow_download:
        raise OhlcvQualityError(
            f"missing cached month {path.name} under {path.parent}; "
            "downloads are off (pass allow_download or use dipcatcher collect)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    url = month_parquet_url(year, month, revision)
    try:
        fetcher(url, partial)
        _require_vendor_parquet(partial)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(path)
    return path


def _require_vendor_parquet(path: Path) -> None:
    try:
        schema = pl.scan_parquet(path).collect_schema()
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise OhlcvQualityError(
            f"cached parquet is unreadable ({path}). Delete it and collect again; it was not repaired."
        ) from exc
    missing = [name for name in VENDOR_COLUMNS if name not in schema]
    if missing:
        raise OhlcvQualityError(f"{path.name} missing vendor columns: {missing}")
    stamp = schema["timestamp"]
    if not isinstance(stamp, pl.Datetime) or stamp.time_zone is None:
        raise OhlcvQualityError(f"{path.name} timestamp must be timezone-aware")


def _scan_vendor(
    paths: list[Path],
    symbols: list[str],
    *,
    start: datetime | None,
    end: datetime | None,
) -> pl.DataFrame:
    lazy = (
        pl.scan_parquet([str(path) for path in paths])
        .select(list(VENDOR_COLUMNS))
        .with_columns(pl.col("ticker").cast(pl.String).str.strip_chars().str.to_uppercase())
        .filter(pl.col("ticker").is_in(symbols))
    )
    if start is not None:
        lazy = lazy.filter(pl.col("timestamp") >= (start - timedelta(minutes=1)))
    if end is not None:
        lazy = lazy.filter(pl.col("timestamp") <= (end - timedelta(minutes=1)))
    return lazy.collect()
