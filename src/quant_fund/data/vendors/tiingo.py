"""Tiingo daily EOD adapter (https://api.tiingo.com).

Wire contract: ``GET /tiingo/daily/{ticker}/prices`` with
``Authorization: Token <key>`` (the key never enters the URL). Response is a
JSON array of daily bars with ``date`` (ISO-8601 day), ``open``/``high``/
``low``/``close``/``volume`` plus ``adj*`` and ``divCash``/``splitFactor``
columns. ``date`` is the trading day; the adapter stamps
``event_time == available_time == US session close`` per the repo's equity
bar convention.

Published free-tier limit: ~500 requests/hour — the default pace
(0.12 req/s ≈ 432/hour) stays under it; raise ``rate_per_second`` for paid
tiers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from quant_fund.data.vendors.base import (
    DailyBarsAdapter,
    require_fields,
    us_session_close,
    utc_day,
)
from quant_fund.data.vendors.errors import VendorResponseError


class TiingoDailyAdapter(DailyBarsAdapter):
    name = "tiingo"
    env_key = "TIINGO_API_KEY"
    base_url = "https://api.tiingo.com"
    revision_id = "TIINGO_VENDOR_ADJ"
    # ~432 req/hour: conservative default under the published 500/hour free cap.
    default_rate_per_second = 0.12
    default_capacity = 2.0

    def build_request(
        self,
        symbol: str,
        start: datetime | None,
        end: datetime | None,
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        ticker = symbol.strip().upper()
        if not ticker:
            raise ValueError("symbol must be non-empty")
        params: dict[str, Any] = {"format": "json"}
        if start is not None:
            params["startDate"] = start.date().isoformat()
        if end is not None:
            params["endDate"] = end.date().isoformat()
        headers = {"Authorization": f"Token {self._require_key()}"}
        return f"/tiingo/daily/{ticker}/prices", params, headers

    def rows_from_payload(self, payload: Any, symbol: str) -> list[dict[str, Any]]:
        if not isinstance(payload, list):
            raise VendorResponseError(
                f"tiingo: expected a JSON array of bars, got {type(payload).__name__}"
            )
        rows: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                raise VendorResponseError(f"tiingo: bar row is not an object: {item!r}")
            require_fields(
                item, ("date", "open", "high", "low", "close", "volume"), vendor="tiingo"
            )
            close_ts = us_session_close(utc_day(item["date"]))
            rows.append(
                {
                    "security_id": symbol.strip().upper(),
                    "event_time": close_ts,
                    "available_time": close_ts,
                    "open": item["open"],
                    "high": item["high"],
                    "low": item["low"],
                    "close": item["close"],
                    "volume": item["volume"],
                }
            )
        return rows
