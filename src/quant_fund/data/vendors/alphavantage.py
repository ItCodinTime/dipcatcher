"""Alpha Vantage daily adapter (TIME_SERIES_DAILY).

Wire contract: ``GET /query?function=TIME_SERIES_DAILY&symbol=X&outputsize=
full&apikey=K`` — this API authenticates only via the ``apikey`` query param
(no header auth on the free tier), so the key is attached as a param and is
never logged or embedded. Response: ``{"Meta Data": {...},
"Time Series (Daily)": {"YYYY-MM-DD": {"1. open": ..., "5. volume": ...}}}``.

Alpha Vantage returns HTTP 200 with a ``Note``/``Information`` payload when
throttled instead of a 429 — detected here and raised as
:class:`VendorRateLimitError`, not parsed as bars.

Published free-tier limit: 25 requests/day (historically 5/min, 500/day —
verify your tier). The default pace (0.08 req/s ≈ 5/min) matches the
long-standing per-minute cap; the daily quota is the caller's budget — pass
a smaller ``rate_per_second`` for strict 25/day compliance.
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
from quant_fund.data.vendors.errors import VendorRateLimitError, VendorResponseError

# Alpha Vantage wire keys inside each daily bar object.
_FIELD_OPEN = "1. open"
_FIELD_HIGH = "2. high"
_FIELD_LOW = "3. low"
_FIELD_CLOSE = "4. close"
_FIELD_VOLUME = "5. volume"


class AlphaVantageDailyAdapter(DailyBarsAdapter):
    name = "alpha_vantage"
    env_key = "ALPHAVANTAGE_API_KEY"
    base_url = "https://www.alphavantage.co"
    revision_id = "ALPHAVANTAGE_VENDOR_ADJ"
    default_rate_per_second = 0.08  # ~5 req/min; see module docstring on daily caps
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
        params: dict[str, Any] = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": "full",
            "datatype": "json",
            "apikey": self._require_key(),
        }
        return "/query", params, {}

    def rows_from_payload(self, payload: Any, symbol: str) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            raise VendorResponseError(
                f"alpha_vantage: expected a JSON object, got {type(payload).__name__}"
            )
        if "Note" in payload or "Information" in payload:
            detail = payload.get("Note") or payload.get("Information")
            raise VendorRateLimitError(
                f"alpha_vantage: throttled by vendor: {str(detail)[:200]}",
                status_code=None,
            )
        if "Error Message" in payload:
            raise VendorResponseError(
                f"alpha_vantage: API error: {str(payload['Error Message'])[:200]}"
            )
        series = payload.get("Time Series (Daily)")
        if series is None:
            raise VendorResponseError(
                "alpha_vantage: payload missing 'Time Series (Daily)' series"
            )
        if not isinstance(series, dict):
            raise VendorResponseError("alpha_vantage: daily series is not an object")
        rows: list[dict[str, Any]] = []
        for day_s, item in series.items():
            if not isinstance(item, dict):
                raise VendorResponseError(
                    f"alpha_vantage: bar row is not an object: {item!r}"
                )
            require_fields(
                item,
                (_FIELD_OPEN, _FIELD_HIGH, _FIELD_LOW, _FIELD_CLOSE, _FIELD_VOLUME),
                vendor="alpha_vantage",
            )
            close_ts = us_session_close(utc_day(day_s))
            rows.append(
                {
                    "security_id": symbol.strip().upper(),
                    "event_time": close_ts,
                    "available_time": close_ts,
                    "open": item[_FIELD_OPEN],
                    "high": item[_FIELD_HIGH],
                    "low": item[_FIELD_LOW],
                    "close": item[_FIELD_CLOSE],
                    "volume": item[_FIELD_VOLUME],
                }
            )
        return rows
