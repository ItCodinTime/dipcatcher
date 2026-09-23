"""Canonical normalization helpers for public data feeds."""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Iterable, Mapping
from typing import Any

import polars as pl

from quant_fund.data.sources.base import SourceError, parse_time, pit_frame


def _number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SourceError(f"{field} is not numeric") from exc
    if not math.isfinite(number):
        raise SourceError(f"{field} is not finite")
    return number


def normalize_ohlcv(
    rows: Iterable[Mapping[str, Any]],
    *,
    source: str,
    revision_id: str = "v1",
    symbol_key: str = "security_id",
) -> pl.DataFrame:
    """Normalize vendor OHLCV rows into Dipcatcher's bar contract."""
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[Any, str]] = set()
    for raw in rows:
        symbol = str(raw.get(symbol_key) or raw.get("symbol") or "").strip()
        if not symbol:
            raise SourceError("OHLCV row has no security_id/symbol")
        event_value = raw.get("event_time", raw.get("timestamp", raw.get("time")))
        if event_value is None:
            raise SourceError("OHLCV row has no event_time/timestamp")
        event = parse_time(event_value)
        key = (event, symbol)
        if key in seen:
            raise SourceError(f"duplicate OHLCV key: {event.isoformat()} {symbol}")
        seen.add(key)
        values = {
            name: _number(raw.get(name), name)
            for name in ("open", "high", "low", "close", "volume")
        }
        if min(values[name] for name in ("open", "high", "low", "close")) <= 0:
            raise SourceError("OHLCV prices must be positive")
        if values["volume"] < 0 or values["low"] > values["high"]:
            raise SourceError("invalid OHLCV envelope or volume")
        if (
            not values["low"] <= values["open"] <= values["high"]
            or not values["low"] <= values["close"] <= values["high"]
        ):
            raise SourceError("open and close must be within low/high")
        normalized.append(
            {
                "security_id": symbol,
                "symbol": symbol,
                **values,
                "event_time": event,
                "available_time": raw.get("available_time"),
            }
        )
    return pit_frame(normalized, source=source, revision_id=revision_id).sort(
        ["security_id", "event_time"]
    )


def normalize_observations(
    rows: Iterable[Mapping[str, Any]],
    *,
    source: str,
    revision_id: str = "v1",
    value_key: str = "value",
    entity_key: str = "security_id",
) -> pl.DataFrame:
    """Normalize dated macro/news/positioning observations."""
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        event_value = raw.get("event_time", raw.get("date", raw.get("timestamp")))
        if event_value is None:
            raise SourceError("observation has no date/event_time")
        value = raw.get(value_key)
        if value is None or value in ("", ".", "null", "NULL"):
            continue
        if raw.get("available_time") is None:
            raise SourceError("observation has no available_time; release time must be explicit")
        row = {
            "security_id": str(
                raw.get(entity_key) or raw.get("series_id") or raw.get("indicator") or source
            ),
            "value": value,
            "event_time": event_value,
            "available_time": raw["available_time"],
        }
        for key in ("unit", "title", "country", "category", "raw_value"):
            if key in raw:
                row[key] = raw[key]
        normalized.append(row)
    return pit_frame(normalized, source=source, revision_id=revision_id).sort(
        ["security_id", "event_time"]
    )


def csv_rows(text: str) -> list[dict[str, str]]:
    """Read a CSV payload without assuming a vendor-specific dataframe library."""
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise SourceError("CSV response has no header")
    return [dict(row) for row in reader]
