"""Strict local normalizer for Binance public daily spot klines.

This adapter consumes already-preserved archives and checksum files.  It never
fetches from the network and labels the resulting observations as candidate-only
because Binance archives can be corrected without historical release vintages.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import zipfile
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import polars as pl

BINANCE_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT")
_SOURCE = "binance-public-data"
_MAX_ARCHIVE_BYTES = 10 * 1024 * 1024
_MAX_MEMBER_BYTES = 1 * 1024 * 1024
_MAX_COMPRESSION_RATIO = 100
_URL_RE = re.compile(
    r"^https://data\.binance\.vision/data/spot/daily/klines/"
    r"(?P<symbol>[A-Z0-9]+)/1d/(?P<filename>[A-Z0-9]+)-1d-(?P<date>\d{4}-\d{2}-\d{2})\.zip$"
)
_SHA_RE = re.compile(r"^(?P<sha>[0-9a-fA-F]{64})\s+(?P<name>\S+)\s*$")


@dataclass(frozen=True)
class BinanceArchiveSpec:
    """One immutable Binance archive receipt to normalize."""

    symbol: str
    date: str
    archive_path: Path
    checksum_path: Path
    url: str
    revision_id: str


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid Binance archive date: {value!r}") from exc


def _validate_spec(spec: BinanceArchiveSpec) -> tuple[date, str]:
    symbol = spec.symbol.strip().upper()
    if spec.symbol != symbol:
        raise ValueError(f"canonical Binance symbol required: {spec.symbol!r}")
    if symbol not in BINANCE_SYMBOLS:
        raise ValueError(f"unexpected Binance symbol: {spec.symbol!r}")
    day = _parse_date(spec.date)
    if not spec.revision_id.strip():
        raise ValueError("revision_id must be non-empty")
    match = _URL_RE.fullmatch(spec.url)
    if match is None:
        raise ValueError(f"invalid official Binance URL: {spec.url!r}")
    if match.group("symbol") != symbol or match.group("date") != spec.date:
        raise ValueError("Binance URL does not match symbol/date")
    expected_name = f"{symbol}-1d-{spec.date}.zip"
    if match.group("filename") != symbol:
        raise ValueError("Binance URL filename does not match symbol/date")
    if spec.archive_path.name != expected_name:
        raise ValueError(f"archive filename must be {expected_name}")
    if not spec.archive_path.is_file() or not spec.checksum_path.is_file():
        raise ValueError("Binance archive and checksum files must exist")
    checksum_line = spec.checksum_path.read_text(encoding="utf-8").strip()
    checksum_match = _SHA_RE.fullmatch(checksum_line)
    if checksum_match is None or checksum_match.group("name") != expected_name:
        raise ValueError("malformed Binance checksum")
    actual = hashlib.sha256(spec.archive_path.read_bytes()).hexdigest()
    if actual.lower() != checksum_match.group("sha").lower():
        raise ValueError(f"sha256 mismatch for {spec.archive_path}")
    return day, actual


def _read_daily_row(
    spec: BinanceArchiveSpec, day: date
) -> tuple[float, float, float, float, float, datetime, int]:
    member_name = f"{spec.symbol}-1d-{spec.date}.csv"
    try:
        if spec.archive_path.stat().st_size > _MAX_ARCHIVE_BYTES:
            raise ValueError("Binance archive exceeds maximum size")
        with zipfile.ZipFile(spec.archive_path) as archive:
            members = archive.namelist()
            if members != [member_name]:
                raise ValueError(f"expected exactly one Binance CSV member {member_name!r}")
            member = archive.getinfo(member_name)
            if member.file_size > _MAX_MEMBER_BYTES:
                raise ValueError("Binance ZIP member exceeds maximum size")
            if member.compress_size == 0 or member.file_size > member.compress_size * _MAX_COMPRESSION_RATIO:
                raise ValueError("Binance ZIP member exceeds maximum compression ratio")
            raw = archive.read(member_name)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"invalid Binance ZIP archive: {spec.archive_path}") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Binance daily archive CSV must be valid UTF-8") from exc
    try:
        rows = list(csv.reader(io.StringIO(text)))
    except csv.Error as exc:
        raise ValueError("Binance daily archive contains malformed CSV") from exc
    if len(rows) != 1 or len(rows[0]) != 12:
        raise ValueError("Binance daily archive must contain exactly one 12-column row")
    values = rows[0]
    try:
        open_ms = int(values[0])
        close_ms = int(values[6])
        open_price, high, low, close, volume = (float(values[i]) for i in (1, 2, 3, 4, 5))
        quote_volume, taker_base_volume, taker_quote_volume = (
            float(values[i]) for i in (7, 9, 10)
        )
        trades = int(values[8])
    except (TypeError, ValueError) as exc:
        raise ValueError("Binance kline row contains malformed numeric values") from exc
    auxiliary = (quote_volume, taker_base_volume, taker_quote_volume)
    if not all(math.isfinite(value) and value >= 0 for value in auxiliary):
        raise ValueError("auxiliary Binance kline values must be finite and non-negative")
    if values[11] != "0":
        raise ValueError("Binance kline ignore column must be 0")
    if trades < 0:
        raise ValueError("Binance number_of_trades must be non-negative")
    if not all(math.isfinite(value) for value in (open_price, high, low, close, volume)):
        raise ValueError("Binance OHLCV values must be finite")
    if min(open_price, high, low, close) <= 0 or volume < 0:
        raise ValueError("Binance prices must be positive and volume non-negative")
    if not low <= open_price <= high or not low <= close <= high or high < low:
        raise ValueError("Binance OHLC envelope is invalid")
    try:
        event = datetime.fromtimestamp(close_ms / 1000, tz=UTC)
        opened = datetime.fromtimestamp(open_ms / 1000, tz=UTC)
    except (OSError, OverflowError, ValueError) as exc:
        raise ValueError("Binance kline timestamp is out of range") from exc
    expected_open = datetime.combine(day, time.min, tzinfo=UTC)
    expected_close = datetime.combine(day, time.max, tzinfo=UTC).replace(microsecond=999000)
    if opened != expected_open or event != expected_close:
        raise ValueError("Binance timestamps do not cover the requested UTC day")
    return open_price, high, low, close, volume, event, trades


def normalize_binance_archives(
    specs: list[BinanceArchiveSpec] | tuple[BinanceArchiveSpec, ...],
    *,
    ingested_time: datetime,
) -> pl.DataFrame:
    """Verify and normalize one common-date five-symbol archive set."""
    if ingested_time.tzinfo is None:
        raise ValueError("ingested_time must be timezone-aware UTC")
    ingested_time = ingested_time.astimezone(UTC)
    if len(specs) != len(BINANCE_SYMBOLS):
        raise ValueError(f"exactly {len(BINANCE_SYMBOLS)} Binance symbols are required")
    seen: set[str] = set()
    revision_ids: set[str] = set()
    rows: list[dict[str, object]] = []
    common_day: date | None = None
    for spec in specs:
        day, _ = _validate_spec(spec)
        symbol = spec.symbol.strip().upper()
        if symbol in seen:
            raise ValueError(f"duplicate Binance symbol: {symbol}")
        seen.add(symbol)
        revision_ids.add(spec.revision_id)
        if common_day is None:
            common_day = day
        elif day != common_day:
            raise ValueError("all Binance archives must share one UTC date")
        open_price, high, low, close, volume, event_time, _ = _read_daily_row(spec, day)
        available_time = datetime.combine(day + timedelta(days=1), time.min, tzinfo=UTC)
        if ingested_time < available_time:
            raise ValueError("ingested_time cannot precede Binance next-day availability")
        rows.append(
            {
                "security_id": f"BINANCE:{symbol}",
                "symbol": symbol,
                "event_time": event_time,
                "available_time": available_time,
                "ingested_time": ingested_time,
                "source": _SOURCE,
                "revision_id": spec.revision_id,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "currency": "USDT",
                "session": "24x7",
            }
        )
    if seen != set(BINANCE_SYMBOLS):
        raise ValueError(f"Binance symbol set must be exactly {BINANCE_SYMBOLS}")
    if len(revision_ids) != 1:
        raise ValueError("all Binance archives must use the same revision_id")
    return pl.DataFrame(rows).select(
        "security_id", "symbol", "event_time", "available_time", "ingested_time",
        "source", "revision_id", "open", "high", "low", "close", "volume",
        "currency", "session",
    ).sort("symbol")
