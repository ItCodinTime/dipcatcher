"""Shared interface for market-data vendor adapters.

One adapter = one vendor endpoint family. ``get_daily_bars`` is the live
path (build request → paced ``VendorClient.get`` → normalize); ``parse_body``
is the offline path used by fixture replay and the reconciliation tool —
it performs the exact same normalization on a recorded response body, so
fixtures exercise real parse logic, not a parallel implementation.

Bar contract: adapters emit the repo's canonical PIT OHLCV frame via
``quant_fund.data.sources.normalize.normalize_ohlcv`` —
``security_id, symbol, open, high, low, close, volume, event_time,
available_time, ingested_time, source, revision_id``, sorted by
``(security_id, event_time)``, duplicates and contract violations failing
closed with ``SourceError``.

PIT convention for equity daily bars matches the Stooq file-tape:
``event_time == available_time == session close`` (16:00 America/New_York
for ``*.us``) — the earliest moment the completed bar could be known.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from typing import Any, Protocol

import polars as pl

from quant_fund.data.adapters.stooq import session_close
from quant_fund.data.sources.base import parse_time, utc_now
from quant_fund.data.sources.normalize import normalize_ohlcv
from quant_fund.data.vendors.client import VendorClient
from quant_fund.data.vendors.errors import VendorAuthError, VendorResponseError

# Schema emitted when a vendor legitimately returns zero bars. Columns mirror
# normalize_ohlcv's output so downstream code can concat without diagonal fill.
EMPTY_BARS_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "security_id": pl.String,
    "symbol": pl.String,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "event_time": pl.Datetime(time_zone="UTC"),
    "available_time": pl.Datetime(time_zone="UTC"),
    "ingested_time": pl.Datetime(time_zone="UTC"),
    "source": pl.String,
    "revision_id": pl.String,
}


def empty_bars_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=EMPTY_BARS_SCHEMA)


def utc_day(value: Any) -> date:
    """Parse a vendor day stamp (ISO str, ms epoch, date/datetime) to a UTC date."""
    if isinstance(value, datetime):
        aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return aware.astimezone(UTC).date()
    if isinstance(value, date):
        return value
    return parse_time(value).date()


def us_session_close(day: date) -> datetime:
    """16:00 America/New_York on ``day``, as aware UTC — the equity bar stamp."""
    return session_close(day, "x.us")


def vendor_session_close(day: date, vendor_symbol: str) -> datetime:
    """Session close keyed on the vendor symbol suffix (``*.us`` / ``*.uk``)."""
    return session_close(day, vendor_symbol)


class VendorBarAdapter(Protocol):
    """One interface for daily-bar vendor adapters (structural)."""

    name: str

    def get_daily_bars(
        self,
        symbol: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame: ...

    def parse_body(
        self,
        body: bytes | str | Any,
        *,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame: ...


class DailyBarsAdapter:
    """Base class: lazy API key, shared client plumbing, shared normalize.

    ``api_key`` resolution is lazy: construction never fails on a missing
    key so adapters can parse fixtures with zero credentials; the first live
    request raises :class:`VendorAuthError`. Explicit ``api_key=`` wins over
    the adapter's ``env_key`` environment variable.
    """

    name = "vendor"
    env_key: str | None = None
    base_url = ""
    revision_id = "VENDOR_DAILY"
    default_rate_per_second = 1.0
    default_capacity: float | None = 1.0
    response_kind = "json"  # "json" | "text"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: VendorClient | None = None,
        transport: Any = None,
        environ: Mapping[str, str] | None = None,
        rate_per_second: float | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
        now: Callable[[], datetime] = utc_now,
        timeout: float = 30.0,
    ) -> None:
        env = environ if environ is not None else os.environ
        raw_key = api_key if api_key is not None else (env.get(self.env_key) if self.env_key else None)
        self._api_key = raw_key.strip() if raw_key and raw_key.strip() else None
        self._now = now
        self._client = client or VendorClient(
            self.base_url,
            rate_per_second=rate_per_second or self.default_rate_per_second,
            capacity=self.default_capacity,
            transport=transport,
            sleeper=sleeper,
            clock=clock,
            rng=rng,
            timeout=timeout,
            now=now,
        )

    # -- per-vendor contract -------------------------------------------------

    def build_request(
        self,
        symbol: str,
        start: datetime | None,
        end: datetime | None,
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """Return ``(url_or_path, params, headers)`` for one daily-bars GET."""
        raise NotImplementedError

    def rows_from_payload(self, payload: Any, symbol: str) -> list[dict[str, Any]]:
        """Convert one vendor payload into pre-normalize OHLCV row dicts."""
        raise NotImplementedError

    # -- shared pipeline ------------------------------------------------------

    def _require_key(self) -> str:
        if self._api_key is None:
            raise VendorAuthError(
                f"{self.name} requires an API key in env var {self.env_key} "
                "(or pass api_key=); credentials are never embedded"
            )
        return self._api_key

    def _decode(self, body: bytes | str | Any) -> Any:
        if self.response_kind == "text":
            if isinstance(body, bytes):
                return body.decode("utf-8-sig", errors="replace")
            if not isinstance(body, str):
                raise VendorResponseError(
                    f"{self.name}: expected text payload, got {type(body).__name__}"
                )
            return body
        if isinstance(body, (bytes, str)):
            try:
                return json.loads(body)
            except json.JSONDecodeError as exc:
                raise VendorResponseError(
                    f"{self.name}: response is not valid JSON"
                ) from exc
        return body  # already-decoded fixture/programmatic payload

    def parse_body(
        self,
        body: bytes | str | Any,
        *,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        """Normalize a recorded/live response body into the repo bar frame."""
        payload = self._decode(body)
        rows = self.rows_from_payload(payload, symbol)
        if not rows:
            return empty_bars_frame()
        frame = normalize_ohlcv(rows, source=self.name, revision_id=self.revision_id)
        return self._filter_window(frame, start, end)

    @staticmethod
    def _filter_window(
        frame: pl.DataFrame,
        start: datetime | None,
        end: datetime | None,
    ) -> pl.DataFrame:
        if start is not None:
            frame = frame.filter(pl.col("event_time") >= start.astimezone(UTC))
        if end is not None:
            frame = frame.filter(pl.col("event_time") <= end.astimezone(UTC))
        return frame

    def _fetch_rows(
        self,
        symbol: str,
        start: datetime | None,
        end: datetime | None,
    ) -> list[dict[str, Any]]:
        """Live path: one GET → pre-normalize rows. Paginated vendors override."""
        url, params, headers = self.build_request(symbol, start, end)
        payload = self._decode(self._client.get_bytes(url, params=params, headers=headers))
        return self.rows_from_payload(payload, symbol)

    def get_daily_bars(
        self,
        symbol: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        rows = self._fetch_rows(symbol, start, end)
        if not rows:
            return empty_bars_frame()
        frame = normalize_ohlcv(rows, source=self.name, revision_id=self.revision_id)
        return self._filter_window(frame, start, end)


def require_fields(row: Mapping[str, Any], fields: tuple[str, ...], *, vendor: str) -> None:
    """Fail closed when a vendor bar is missing required keys."""
    missing = [name for name in fields if row.get(name) is None]
    if missing:
        raise VendorResponseError(f"{vendor}: bar row missing fields {missing}: {row!r}")
