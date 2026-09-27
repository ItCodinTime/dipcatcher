"""Licensed-vendor HTTP market data adapter skeleton. Fail-closed, PIT-first.

This module implements the ``MarketDataProvider`` protocol for a licensed,
point-in-time vendor feed. It is an *interface*, not an entitlement: the
adapter refuses to construct unless (a) an API key is present in the
configured environment variable and (b) the caller explicitly sets
``license_acknowledged=True`` in :class:`VendorHttpConfig`. No entitlement,
no object — mirroring the lab rule that missing configuration fails closed.

Point-in-time contract: every normalized row carries ``release_ts`` (when the
vendor released the record) and ``ingest_ts`` (when the record was ingested),
both timezone-aware UTC. Rows missing either are rejected, never silently
filled. Duplicates on ``(security_id, ts)``, non-finite OHLCV, non-positive
prices, negative volume, and inverted envelopes fail closed with
``VendorDataError`` — same honesty contract as the Stooq/Yahoo file-tape
adapters, but strict (a licensed feed breaching its contract is a data
incident, not a row to skip).

No network at import time. The HTTP layer is an injectable ``Transport``
callable so tests never touch the network; the default pooled HTTP transport is
only built when the caller does not inject one and a request is actually made.

This is still NOT a live data source and confers no vendor entitlement; see
``docs/INSTITUTIONAL_READINESS.md``.
"""

from __future__ import annotations

import json
import os
import urllib.parse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import polars as pl

from quant_fund.data.adapters.stooq import USER_AGENT, np_finite
from quant_fund.schemas.errors import ConfigError, DataContractError

SOURCE = "vendor_http"
REVISION = "VENDOR_HTTP_LICENSED"
DEFAULT_API_KEY_ENV = "VENDOR_HTTP_API_KEY"

Transport = Callable[[str, Mapping[str, str]], bytes]
"""Injectable HTTP GET: ``(url, headers) -> response body``. Never module-level."""


class VendorEntitlementError(ConfigError):
    """Missing entitlement: no API key or license not acknowledged."""


class VendorDataError(DataContractError):
    """Vendor payload breached the point-in-time/OHLCV data contract."""


BARS_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "security_id": pl.String,
    "symbol": pl.String,
    "event_time": pl.Datetime(time_zone="UTC"),
    "available_time": pl.Datetime(time_zone="UTC"),
    "ingested_time": pl.Datetime(time_zone="UTC"),
    "release_ts": pl.Datetime(time_zone="UTC"),
    "ingest_ts": pl.Datetime(time_zone="UTC"),
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

ACTIONS_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "security_id": pl.String,
    "event_time": pl.Datetime(time_zone="UTC"),
    "available_time": pl.Datetime(time_zone="UTC"),
    "ingested_time": pl.Datetime(time_zone="UTC"),
    "release_ts": pl.Datetime(time_zone="UTC"),
    "ingest_ts": pl.Datetime(time_zone="UTC"),
    "source": pl.String,
    "revision_id": pl.String,
    "action_type": pl.String,
    "factor": pl.Float64,
    "amount": pl.Float64,
    "new_ticker": pl.String,
}

MASTER_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "security_id": pl.String,
    "ticker": pl.String,
    "name": pl.String,
    "exchange": pl.String,
    "currency": pl.String,
    "sector": pl.String,
    "industry": pl.String,
    "security_type": pl.String,
    "valid_from": pl.Datetime(time_zone="UTC"),
    "valid_to": pl.Datetime(time_zone="UTC"),
    "available_time": pl.Datetime(time_zone="UTC"),
    "ingested_time": pl.Datetime(time_zone="UTC"),
    "release_ts": pl.Datetime(time_zone="UTC"),
    "ingest_ts": pl.Datetime(time_zone="UTC"),
    "source": pl.String,
    "revision_id": pl.String,
}


@dataclass(frozen=True)
class VendorHttpConfig:
    """Entitlement + endpoint config. ``license_acknowledged`` must be True."""

    vendor: str
    base_url: str
    license_acknowledged: bool
    api_key_env: str = DEFAULT_API_KEY_ENV
    revision_id: str = REVISION
    timeout_s: float = 30.0
    environ: Mapping[str, str] | None = None
    """Optional environment override (tests); defaults to ``os.environ``."""


def _default_transport(timeout_s: float) -> Transport:
    """Pooled HTTP transport. Built lazily; never invoked at import time."""
    from quant_fund.data.sources.base import HttpClient

    client = HttpClient(timeout=timeout_s, retries=2)

    def _fetch(url: str, headers: Mapping[str, str]) -> bytes:
        return client.get_bytes(url, headers=dict(headers))

    return _fetch


def _parse_utc(value: object, *, field: str) -> datetime:
    """Parse an ISO-8601 timestamp; reject missing/naive values (fail closed)."""
    if not isinstance(value, str) or not value.strip():
        raise VendorDataError(f"{field} must be a non-empty ISO-8601 timestamp, got {value!r}")
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise VendorDataError(f"{field} is not ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        raise VendorDataError(f"{field} must be timezone-aware (UTC required): {value!r}")
    return parsed.astimezone(UTC)


def _require_str(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise VendorDataError(f"{field} must be a non-empty string, got {value!r}")
    return value.strip()


def _opt_str(row: Mapping[str, object], field: str) -> str | None:
    value = row.get(field)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _finite_float(value: object, *, field: str) -> float:
    try:
        out = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise VendorDataError(f"{field} must be numeric, got {value!r}") from exc
    if not np_finite(out):
        raise VendorDataError(f"{field} must be finite, got {out}")
    return out


def _decode_json(body: bytes, *, endpoint: str) -> dict[str, object]:
    try:
        payload: object = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VendorDataError(f"{endpoint}: response is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise VendorDataError(f"{endpoint}: payload is not a JSON object")
    return cast(dict[str, object], payload)


def _payload_rows(payload: Mapping[str, object], key: str, *, endpoint: str) -> list[object]:
    rows = payload.get(key)
    if not isinstance(rows, list):
        raise VendorDataError(f"{endpoint}: payload missing '{key}' list")
    return cast(list[object], rows)


def _as_row(raw: object, *, endpoint: str) -> Mapping[str, object]:
    if not isinstance(raw, dict):
        raise VendorDataError(f"{endpoint}: row is not a JSON object: {raw!r}")
    return cast(Mapping[str, object], raw)


class VendorHttpAdapter:
    """``MarketDataProvider`` over a licensed vendor HTTP API. Fail-closed."""

    def __init__(
        self,
        config: VendorHttpConfig,
        *,
        transport: Transport | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not config.license_acknowledged:
            raise VendorEntitlementError(
                "license_acknowledged must be True; no vendor entitlement is implied"
            )
        environ = config.environ if config.environ is not None else os.environ
        api_key = environ.get(config.api_key_env, "").strip()
        if not api_key:
            raise VendorEntitlementError(
                f"missing API key in environment variable {config.api_key_env}; "
                "refusing to construct without an entitlement"
            )
        if not config.base_url.startswith(("https://", "http://")):
            raise VendorEntitlementError(f"base_url must be an HTTP(S) URL: {config.base_url!r}")
        self._config = config
        self._headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }
        if transport is not None:
            self._transport = transport
        else:
            self._transport = _default_transport(config.timeout_s)
        self._clock = clock if clock is not None else (lambda: datetime.now(tz=UTC))

    # -- HTTP layer (only reached via get_*; injectable, never at import) --

    def _get(self, endpoint: str, params: list[tuple[str, str]]) -> dict[str, object]:
        url = f"{self._config.base_url.rstrip('/')}/{endpoint}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        return _decode_json(self._transport(url, self._headers), endpoint=endpoint)

    def _check_pit(self, ts: datetime, release_ts: datetime, ingest_ts: datetime) -> None:
        if release_ts > ingest_ts:
            raise VendorDataError(f"release_ts {release_ts} is after ingest_ts {ingest_ts}")
        if ts > ingest_ts:
            raise VendorDataError(f"event ts {ts} is after ingest_ts {ingest_ts}")
        if ingest_ts > self._clock():
            raise VendorDataError(f"ingest_ts {ingest_ts} is in the future")

    def _pit_fields(self, row: Mapping[str, object]) -> tuple[datetime, datetime, datetime]:
        ts = _parse_utc(row.get("ts"), field="ts")
        release_ts = _parse_utc(row.get("release_ts"), field="release_ts")
        ingest_ts = _parse_utc(row.get("ingest_ts"), field="ingest_ts")
        self._check_pit(ts, release_ts, ingest_ts)
        return ts, release_ts, ingest_ts

    # -- MarketDataProvider protocol --

    def get_bars(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        security_ids: list[str] | None = None,
    ) -> pl.DataFrame:
        params: list[tuple[str, str]] = []
        if start is not None:
            params.append(("start", start.astimezone(UTC).isoformat()))
        if end is not None:
            params.append(("end", end.astimezone(UTC).isoformat()))
        if security_ids:
            params.append(("symbols", ",".join(security_ids)))
        payload = self._get("bars", params)
        frame = self._normalize_bars(_payload_rows(payload, "bars", endpoint="bars"))
        if start is not None and not frame.is_empty():
            frame = frame.filter(pl.col("event_time") >= start)
        if end is not None and not frame.is_empty():
            frame = frame.filter(pl.col("event_time") <= end)
        if security_ids and not frame.is_empty():
            frame = frame.filter(pl.col("security_id").is_in(security_ids))
        return frame.sort(["event_time", "security_id"])

    def get_corporate_actions(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        params: list[tuple[str, str]] = []
        if start is not None:
            params.append(("start", start.astimezone(UTC).isoformat()))
        if end is not None:
            params.append(("end", end.astimezone(UTC).isoformat()))
        payload = self._get("corporate_actions", params)
        frame = self._normalize_actions(
            _payload_rows(payload, "corporate_actions", endpoint="corporate_actions")
        )
        if start is not None and not frame.is_empty():
            frame = frame.filter(pl.col("event_time") >= start)
        if end is not None and not frame.is_empty():
            frame = frame.filter(pl.col("event_time") <= end)
        return frame.sort(["event_time", "security_id"])

    def get_security_master(self) -> pl.DataFrame:
        payload = self._get("securities", [])
        return self._normalize_master(_payload_rows(payload, "securities", endpoint="securities"))

    # -- Normalization + validation (fail closed, never silently filled) --

    def _normalize_bars(self, rows: list[object]) -> pl.DataFrame:
        out: list[dict[str, object]] = []
        seen: set[tuple[str, datetime]] = set()
        for raw in rows:
            row = _as_row(raw, endpoint="bars")
            security_id = _require_str(row, "security_id")
            ts, release_ts, ingest_ts = self._pit_fields(row)
            key = (security_id, ts)
            if key in seen:
                raise VendorDataError(f"duplicate bar for (security_id, ts) = {key}")
            seen.add(key)
            opn = _finite_float(row.get("open"), field="open")
            high = _finite_float(row.get("high"), field="high")
            low = _finite_float(row.get("low"), field="low")
            close = _finite_float(row.get("close"), field="close")
            volume = _finite_float(row.get("volume"), field="volume")
            if not all(v > 0 for v in (opn, high, low, close)):
                raise VendorDataError(f"OHLC must be positive for {key}")
            if volume < 0:
                raise VendorDataError(f"negative volume for {key}")
            if high < low:
                raise VendorDataError(f"inverted envelope (high < low) for {key}")
            out.append(
                {
                    "security_id": security_id,
                    "symbol": security_id,
                    "event_time": ts,
                    "available_time": release_ts,
                    "ingested_time": ingest_ts,
                    "release_ts": release_ts,
                    "ingest_ts": ingest_ts,
                    "source": SOURCE,
                    "revision_id": self._config.revision_id,
                    "open": opn,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "currency": _opt_str(row, "currency") or "USD",
                    "session": _opt_str(row, "session") or "rth",
                }
            )
        if not out:
            return pl.DataFrame(schema=BARS_SCHEMA)
        return pl.DataFrame(out)

    def _normalize_actions(self, rows: list[object]) -> pl.DataFrame:
        out: list[dict[str, object]] = []
        seen: set[tuple[str, datetime, str]] = set()
        for raw in rows:
            row = _as_row(raw, endpoint="corporate_actions")
            security_id = _require_str(row, "security_id")
            action_type = _require_str(row, "action_type")
            ts, release_ts, ingest_ts = self._pit_fields(row)
            key = (security_id, ts, action_type)
            if key in seen:
                raise VendorDataError(f"duplicate corporate action for {key}")
            seen.add(key)
            factor_raw = row.get("factor")
            amount_raw = row.get("amount")
            new_ticker_raw = row.get("new_ticker")
            out.append(
                {
                    "security_id": security_id,
                    "event_time": ts,
                    "available_time": release_ts,
                    "ingested_time": ingest_ts,
                    "release_ts": release_ts,
                    "ingest_ts": ingest_ts,
                    "source": SOURCE,
                    "revision_id": self._config.revision_id,
                    "action_type": action_type,
                    "factor": (
                        1.0 if factor_raw is None else _finite_float(factor_raw, field="factor")
                    ),
                    "amount": (
                        0.0 if amount_raw is None else _finite_float(amount_raw, field="amount")
                    ),
                    "new_ticker": new_ticker_raw if isinstance(new_ticker_raw, str) else None,
                }
            )
        if not out:
            return pl.DataFrame(schema=ACTIONS_SCHEMA)
        return pl.DataFrame(out)

    def _normalize_master(self, rows: list[object]) -> pl.DataFrame:
        out: list[dict[str, object]] = []
        seen: set[str] = set()
        for raw in rows:
            row = _as_row(raw, endpoint="securities")
            security_id = _require_str(row, "security_id")
            ticker = _require_str(row, "ticker")
            if security_id in seen:
                raise VendorDataError(f"duplicate security_id in master: {security_id}")
            seen.add(security_id)
            valid_from = _parse_utc(row.get("valid_from"), field="valid_from")
            release_ts = _parse_utc(row.get("release_ts"), field="release_ts")
            ingest_ts = _parse_utc(row.get("ingest_ts"), field="ingest_ts")
            self._check_pit(valid_from, release_ts, ingest_ts)
            out.append(
                {
                    "security_id": security_id,
                    "ticker": ticker,
                    "name": _opt_str(row, "name") or ticker,
                    "exchange": _opt_str(row, "exchange") or "XNYS",
                    "currency": _opt_str(row, "currency") or "USD",
                    "sector": _opt_str(row, "sector") or "Unknown",
                    "industry": _opt_str(row, "industry") or "Unknown",
                    "security_type": _opt_str(row, "security_type") or "common_stock",
                    "valid_from": valid_from,
                    "valid_to": None,
                    "available_time": release_ts,
                    "ingested_time": ingest_ts,
                    "release_ts": release_ts,
                    "ingest_ts": ingest_ts,
                    "source": SOURCE,
                    "revision_id": self._config.revision_id,
                }
            )
        if not out:
            return pl.DataFrame(schema=MASTER_SCHEMA)
        return pl.DataFrame(out)
