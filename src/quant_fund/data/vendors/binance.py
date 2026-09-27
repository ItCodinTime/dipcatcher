"""Binance public klines adapter — keyless crypto OHLCV.

Wire contract: ``GET /api/v3/klines?symbol=X&interval=1d&startTime=&endTime=
&limit=`` returns a JSON array of kline arrays ``[openTime, o, h, l, c, vol,
closeTime, ...]``. ``event_time`` is the bar open time, ``available_time``
the bar close time (earliest moment the completed bar is knowable) — the
same PIT convention as ``sources.adapters.BinancePublicDataSource``, which
this adapter reimplements behind the ``VendorClient``/httpx interface for
cross-vendor parity. The still-open bar (closeTime in the future) is
dropped.

No API key: public endpoint. Rate limit is weight-based (klines weight 1–10
by ``limit``; ~6000 weight/min IP cap) — the 2 req/s default stays well
under it at any ``limit``.

``get_daily_bars`` fetches ``interval=1d``; ``get_klines`` exposes other
intervals. Pagination walks ``startTime`` forward past each page's last
open time, bounded by ``max_pages``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import polars as pl

from quant_fund.data.sources.normalize import normalize_ohlcv
from quant_fund.data.vendors.base import DailyBarsAdapter, empty_bars_frame
from quant_fund.data.vendors.errors import VendorResponseError

BINANCE_INTERVALS = frozenset(
    {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}
)


class BinanceKlinesAdapter(DailyBarsAdapter):
    name = "binance_klines"
    env_key = None  # keyless public endpoint
    base_url = "https://api.binance.com"
    revision_id = "BINANCE_RAW"
    default_rate_per_second = 2.0
    default_capacity = 5.0
    default_interval = "1d"
    default_limit = 1000
    default_max_pages = 5

    def build_request(
        self,
        symbol: str,
        start: datetime | None,
        end: datetime | None,
        *,
        interval: str | None = None,
        limit: int | None = None,
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        ticker = symbol.strip().upper()
        if not ticker:
            raise ValueError("symbol must be non-empty")
        use_interval = interval or self.default_interval
        if use_interval not in BINANCE_INTERVALS:
            raise ValueError(f"unsupported Binance interval {use_interval!r}")
        use_limit = int(limit if limit is not None else self.default_limit)
        if not 1 <= use_limit <= 1500:
            raise ValueError("Binance limit must be between 1 and 1500")
        params: dict[str, Any] = {
            "symbol": ticker,
            "interval": use_interval,
            "limit": use_limit,
        }
        if start is not None:
            params["startTime"] = int(start.timestamp() * 1000)
        if end is not None:
            params["endTime"] = int(end.timestamp() * 1000)
        return "/api/v3/klines", params, {}

    def _fetch_rows(
        self,
        symbol: str,
        start: datetime | None,
        end: datetime | None,
        *,
        interval: str | None = None,
        limit: int | None = None,
        max_pages: int | None = None,
    ) -> list[dict[str, Any]]:
        """Paginate forward from ``start`` (or the vendor default window)."""
        if max_pages is not None and max_pages < 1:
            raise ValueError("max_pages must be >= 1")
        pages = int(max_pages if max_pages is not None else self.default_max_pages)
        url, params, headers = self.build_request(
            symbol, start, end, interval=interval, limit=limit
        )
        use_limit = int(params["limit"])
        rows: list[dict[str, Any]] = []
        for _ in range(pages):
            payload = self._decode(
                self._client.get_bytes(url, params=params, headers=headers)
            )
            page_rows = self.rows_from_payload(payload, symbol)
            if not page_rows:
                break
            rows.extend(page_rows)
            if not isinstance(payload, list) or len(payload) < use_limit:
                break
            last_open_ms = int(payload[-1][0])
            next_cursor = last_open_ms + 1
            if "endTime" in params and next_cursor >= int(params["endTime"]):
                break
            params = dict(params)
            params["startTime"] = next_cursor
        return rows

    def get_klines(
        self,
        symbol: str,
        *,
        interval: str = "1d",
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
        max_pages: int | None = None,
    ) -> pl.DataFrame:
        rows = self._fetch_rows(
            symbol,
            start,
            end,
            interval=interval,
            limit=limit,
            max_pages=max_pages,
        )
        if not rows:
            return empty_bars_frame()
        frame = normalize_ohlcv(
            rows, source=self.name, revision_id=f"{self.revision_id}.{interval}"
        )
        return self._filter_window(frame, start, end)

    def get_daily_bars(
        self,
        symbol: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        return self.get_klines(symbol, interval=self.default_interval, start=start, end=end)

    def parse_body(
        self,
        body: bytes | str | Any,
        *,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        payload = self._decode(body)
        rows = self.rows_from_payload(payload, symbol)
        if not rows:
            return empty_bars_frame()
        frame = normalize_ohlcv(
            rows,
            source=self.name,
            revision_id=f"{self.revision_id}.{self.default_interval}",
        )
        return self._filter_window(frame, start, end)

    def rows_from_payload(self, payload: Any, symbol: str) -> list[dict[str, Any]]:
        if not isinstance(payload, list):
            raise VendorResponseError(
                f"binance_klines: expected a JSON array, got {type(payload).__name__}"
            )
        if payload and not all(
            isinstance(item, list) and len(item) >= 7 for item in payload
        ):
            raise VendorResponseError("binance_klines: malformed kline row")
        now_ms = int(self._now().timestamp() * 1000)
        rows: list[dict[str, Any]] = []
        for item in payload:
            open_ms, close_ms = int(item[0]), int(item[6])
            if close_ms > now_ms:
                continue  # still-open bar: OHLCV would keep mutating
            rows.append(
                {
                    "security_id": symbol.strip().upper(),
                    "event_time": open_ms,
                    "available_time": close_ms,
                    "open": item[1],
                    "high": item[2],
                    "low": item[3],
                    "close": item[4],
                    "volume": item[5],
                }
            )
        return rows
