"""Polygon.io daily aggregates adapter (v2 aggs endpoint).

Wire contract: ``GET /v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}`` with
``Authorization: Bearer <key>`` (Bearer keeps the key out of the URL; the
``apiKey`` query param equivalent is not used). ``adjusted=true`` requests
split-adjusted bars, matching the repo's vendor-adjusted convention.
Response: ``{"results": [{"t": ms_epoch, "o","h","l","c","v","vw","n"}, ...],
"status": "OK", "resultsCount": N}`` — ``t`` is the aggregate window start
(ms epoch); the adapter stamps ``event_time == available_time == US session
close`` for the bar's trading day.

Published free-tier limit: 5 requests/minute — the default pace
(5/60 req/s) matches it.
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


class PolygonDailyAdapter(DailyBarsAdapter):
    name = "polygon"
    env_key = "POLYGON_API_KEY"
    base_url = "https://api.polygon.io"
    revision_id = "POLYGON_VENDOR_ADJ"
    default_rate_per_second = 5.0 / 60.0  # published free tier: 5 req/min
    default_capacity = 1.0

    def build_request(
        self,
        symbol: str,
        start: datetime | None,
        end: datetime | None,
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        ticker = symbol.strip().upper()
        if not ticker:
            raise ValueError("symbol must be non-empty")
        start_s = (start.date().isoformat() if start is not None else "1900-01-01")
        end_s = end.date().isoformat() if end is not None else "2999-12-31"
        params: dict[str, Any] = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
        }
        headers = {"Authorization": f"Bearer {self._require_key()}"}
        path = f"/v2/aggs/ticker/{ticker}/range/1/day/{start_s}/{end_s}"
        return path, params, headers

    def rows_from_payload(self, payload: Any, symbol: str) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            raise VendorResponseError(
                f"polygon: expected a JSON object, got {type(payload).__name__}"
            )
        if "error" in payload:
            raise VendorResponseError(f"polygon: API error payload: {payload.get('error')!r}")
        results = payload.get("results")
        if results is None:
            # Legit empty range: status OK/DELAYED with resultsCount 0.
            if payload.get("resultsCount", 0) == 0:
                return []
            raise VendorResponseError("polygon: payload missing 'results' list")
        if not isinstance(results, list):
            raise VendorResponseError("polygon: 'results' is not a list")
        rows: list[dict[str, Any]] = []
        for item in results:
            if not isinstance(item, dict):
                raise VendorResponseError(f"polygon: bar row is not an object: {item!r}")
            require_fields(item, ("t", "o", "h", "l", "c", "v"), vendor="polygon")
            close_ts = us_session_close(utc_day(item["t"]))
            rows.append(
                {
                    "security_id": symbol.strip().upper(),
                    "event_time": close_ts,
                    "available_time": close_ts,
                    "open": item["o"],
                    "high": item["h"],
                    "low": item["l"],
                    "close": item["c"],
                    "volume": item["v"],
                }
            )
        return rows
