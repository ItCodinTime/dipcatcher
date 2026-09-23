"""Concrete collectors for the requested public/open data source families."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import polars as pl

from quant_fund.data.sources.base import (
    SourceAdapter,
    SourceError,
    pit_frame,
    query_url,
    utc_now,
)
from quant_fund.data.sources.normalize import csv_rows, normalize_observations, normalize_ohlcv

# Binance USDⓈ-M futures launched September 2019; no perp kline predates this.
PERP_EARLIEST_MS = 1567296000000  # 2019-09-01T00:00:00Z

_BINANCE_INTERVALS = frozenset(
    {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}
)


def _paginated_klines(
    client: Any,
    endpoint: str,
    *,
    symbol: str,
    interval: str,
    start_time: int | None,
    end_time: int | None,
    limit: int,
    max_pages: int,
    pause_seconds: float,
) -> list[list[Any]]:
    """Fetch every kline page for [start_time, end_time) in ascending order.

    Binance klines return at most ``limit`` rows per call starting at
    ``startTime``; without a startTime the API returns only the *most recent*
    bars, so full-history callers must paginate forward explicitly. The cursor
    advances to ``last_open_time + 1`` each page, which guarantees progress even
    when a page is truncated mid-interval. ``max_pages`` bounds total requests.
    """
    if interval not in _BINANCE_INTERVALS:
        raise ValueError(f"unsupported Binance interval {interval!r}")
    if not 1 <= limit <= 1500:
        raise ValueError("Binance limit must be between 1 and 1500")
    if max_pages < 1:
        raise ValueError("max_pages must be >= 1")
    if pause_seconds < 0:
        raise ValueError("pause_seconds must be non-negative")
    cursor = int(start_time) if start_time is not None else PERP_EARLIEST_MS
    rows: list[list[Any]] = []
    for _ in range(max_pages):
        payload = client.get_json(
            query_url(
                endpoint,
                {
                    "symbol": symbol.upper(),
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": end_time,
                    "limit": limit,
                },
            )
        )
        if not isinstance(payload, list):
            raise SourceError("Binance klines response must be a list")
        if not payload:
            break
        rows.extend(payload)
        next_cursor = int(payload[-1][0]) + 1
        if len(payload) < limit or (end_time is not None and next_cursor >= int(end_time)):
            break
        cursor = next_cursor
        if pause_seconds > 0:
            time.sleep(pause_seconds)
    return rows


class BinancePublicDataSource(SourceAdapter):
    name = "binance_public_data"
    endpoint = "https://api.binance.com/api/v3/klines"

    def fetch(
        self,
        *,
        symbol: str = "BTCUSDT",
        interval: str = "1d",
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1000,
        max_pages: int = 1,
        pause_seconds: float = 0.3,
    ) -> pl.DataFrame:
        if not 1 <= limit <= 1000:
            raise ValueError("Binance limit must be between 1 and 1000")
        if start_time is None and max_pages <= 1:
            # Legacy single-page semantics: no startTime -> most recent bars.
            payload = self.client.get_json(
                query_url(
                    self.endpoint,
                    {
                        "symbol": symbol.upper(),
                        "interval": interval,
                        "startTime": None,
                        "endTime": end_time,
                        "limit": limit,
                    },
                )
            )
        else:
            payload = _paginated_klines(
                self.client,
                self.endpoint,
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
                max_pages=max_pages,
                pause_seconds=pause_seconds,
            )
        if not isinstance(payload, list):
            raise SourceError("Binance klines response must be a list")
        if not all(isinstance(item, list) and len(item) >= 7 for item in payload):
            raise SourceError("malformed Binance kline row")
        now_ms = int(utc_now().timestamp() * 1000)
        rows = [
            {
                "security_id": symbol.upper(),
                "event_time": item[0],
                "open": item[1],
                "high": item[2],
                "low": item[3],
                "close": item[4],
                "volume": item[5],
                # Binance kline close time is the earliest conservative availability.
                "available_time": item[6],
            }
            for item in payload
            # Drop the still-open bar: its close/availability time is in the
            # future and its OHLCV values would keep mutating.
            if int(item[6]) <= now_ms
        ]
        if not rows:
            raise SourceError("Binance returned only an in-progress kline")
        return normalize_ohlcv(rows, source=self.name, revision_id=interval)


class BinanceMarketSource(BinancePublicDataSource):
    """REST history plus pure normalization for Binance public WebSocket events."""

    name = "binance_market_websocket"

    @staticmethod
    def normalize_trade(message: dict[str, Any]) -> pl.DataFrame:
        data = message.get("data", message)
        if not isinstance(data, dict) or data.get("e") not in {"trade", "aggTrade"}:
            raise SourceError("unsupported Binance trade message")
        price = data.get("p")
        quantity = data.get("q")
        row = {
            "security_id": data.get("s"),
            "event_time": data.get("T", data.get("E")),
            "available_time": data.get("E", data.get("T")),
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": quantity,
        }
        return normalize_ohlcv(
            [row], source=BinanceMarketSource.name, revision_id=str(data.get("a", "stream"))
        )


class BinanceUsdtmPerpSource(SourceAdapter):
    """Binance USDⓈ-M perpetual klines with full-history pagination.

    Unlike the spot adapter (single ≤1000-bar page), ``fetch`` pages forward
    from ``start_time`` (default: futures launch epoch) so multi-year hourly
    histories arrive complete. The still-open bar is dropped exactly like the
    spot source — its close time lies in the future and its OHLCV mutates.
    """

    name = "binance_usdtm_perp"
    endpoint = "https://fapi.binance.com/fapi/v1/klines"

    def fetch(
        self,
        *,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1500,
        max_pages: int = 200,
        pause_seconds: float = 0.3,
    ) -> pl.DataFrame:
        now_ms = int(utc_now().timestamp() * 1000)
        payload = _paginated_klines(
            self.client,
            self.endpoint,
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            max_pages=max_pages,
            pause_seconds=pause_seconds,
        )
        if payload and not all(isinstance(item, list) and len(item) >= 7 for item in payload):
            raise SourceError("malformed Binance perp kline row")
        rows = [
            {
                "security_id": symbol.upper(),
                "event_time": item[0],
                "open": item[1],
                "high": item[2],
                "low": item[3],
                "close": item[4],
                "volume": item[5],
                "available_time": item[6],
            }
            for item in payload
            if int(item[6]) <= now_ms
        ]
        if not rows:
            raise SourceError("Binance perp returned only an in-progress kline")
        return normalize_ohlcv(rows, source=self.name, revision_id=f"{interval}.usdtm")


class BinanceFundingRateSource(SourceAdapter):
    """Binance USDⓈ-M funding-rate history (charged every 8h or 4h per symbol).

    The realized rate is only fixed at ``fundingTime``, so both event_time and
    available_time are stamped there — never earlier. Rows carry ``value`` =
    funding rate (decimal) and ``mark_price`` for auditability.
    """

    name = "binance_funding_rate"
    endpoint = "https://fapi.binance.com/fapi/v1/fundingRate"

    def fetch(
        self,
        *,
        symbol: str = "BTCUSDT",
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1000,
        max_pages: int = 200,
        pause_seconds: float = 0.3,
    ) -> pl.DataFrame:
        if not 1 <= limit <= 1000:
            raise ValueError("Binance funding limit must be between 1 and 1000")
        if max_pages < 1:
            raise ValueError("max_pages must be >= 1")
        cursor = int(start_time) if start_time is not None else PERP_EARLIEST_MS
        now_ms = int(utc_now().timestamp() * 1000)
        normalized_rows: list[dict[str, Any]] = []
        for _ in range(max_pages):
            payload = self.client.get_json(
                query_url(
                    self.endpoint,
                    {
                        "symbol": symbol.upper(),
                        "startTime": cursor,
                        "endTime": end_time,
                        "limit": limit,
                    },
                )
            )
            if not isinstance(payload, list):
                raise SourceError("Binance funding response must be a list")
            if not payload:
                break
            for item in payload:
                if not isinstance(item, dict):
                    raise SourceError("malformed Binance funding row")
                if int(item["fundingTime"]) > now_ms:
                    # Not yet charged — the rate is only realized at fundingTime.
                    continue
                rate = float(item["fundingRate"])
                if not (rate == rate and abs(rate) < 10.0):
                    raise SourceError("funding rate is not finite")
                normalized_rows.append(
                    {
                        "security_id": str(item.get("symbol", symbol)).upper(),
                        "event_time": item["fundingTime"],
                        "available_time": item["fundingTime"],
                        "value": rate,
                        "raw_value": json.dumps(item, sort_keys=True),
                    }
                )
            next_cursor = int(payload[-1]["fundingTime"]) + 1
            if len(payload) < limit or (end_time is not None and next_cursor >= int(end_time)):
                break
            cursor = next_cursor
            if pause_seconds > 0:
                time.sleep(pause_seconds)
        if not normalized_rows:
            raise SourceError("Binance funding returned no rows")
        return normalize_observations(normalized_rows, source=self.name, revision_id="v1")


class BinancePerpUniverseSource(SourceAdapter):
    """Current USDⓈ-M perpetual universe: listing dates + 24h quote-volume rank.

    The public API only exposes *currently traded* symbols — this snapshot is a
    survivorship-truncated universe and is stamped ``available_time = now`` so
    downstream PIT logic can never pretend it was known earlier.
    """

    name = "binance_perp_universe"
    info_endpoint = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    ticker_endpoint = "https://fapi.binance.com/fapi/v1/ticker/24hr"

    def fetch(
        self,
        *,
        quote_asset: str = "USDT",
        top_n: int | None = None,
        min_quote_volume: float = 0.0,
    ) -> pl.DataFrame:
        if top_n is not None and top_n < 1:
            raise ValueError("top_n must be >= 1")
        info = self.client.get_json(self.info_endpoint)
        tickers = self.client.get_json(self.ticker_endpoint)
        symbols = info.get("symbols") if isinstance(info, dict) else None
        if not isinstance(symbols, list):
            raise SourceError("Binance exchangeInfo response has no symbols list")
        if not isinstance(tickers, list):
            raise SourceError("Binance 24hr ticker response must be a list")
        quote = quote_asset.upper()
        quote_volume = {}
        for item in tickers:
            if isinstance(item, dict) and "symbol" in item:
                try:
                    quote_volume[str(item["symbol"]).upper()] = float(item.get("quoteVolume", 0.0))
                except (TypeError, ValueError):
                    continue
        now = utc_now()
        rows: list[dict[str, Any]] = []
        for item in symbols:
            if not isinstance(item, dict):
                continue
            if (
                item.get("contractType") != "PERPETUAL"
                or str(item.get("quoteAsset", "")).upper() != quote
                or item.get("status") != "TRADING"
            ):
                continue
            symbol = str(item.get("symbol", "")).upper()
            if not symbol:
                continue
            onboard = item.get("onboardDate")
            onboard_ms = int(onboard) if onboard is not None else PERP_EARLIEST_MS
            volume = quote_volume.get(symbol)
            if volume is None or volume < float(min_quote_volume):
                continue
            rows.append(
                {
                    "security_id": symbol,
                    "event_time": onboard_ms,
                    "available_time": now,
                    "value": volume,
                    "base_asset": str(item.get("baseAsset", "")),
                    "onboard_ms": onboard_ms,
                }
            )
        rows.sort(key=lambda r: (-float(r["value"]), r["security_id"]))
        if top_n is not None:
            rows = rows[:top_n]
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        if not rows:
            raise SourceError("Binance perp universe resolved to zero symbols")
        return pit_frame(rows, source=self.name, revision_id="v1").sort(["rank"])


class GdeltSource(SourceAdapter):
    name = "gdelt"
    endpoint = "https://api.gdeltproject.org/api/v2/doc/doc"

    def fetch(
        self, *, query: str, mode: str = "timelinevol", format: str = "json", maxrecords: int = 250
    ) -> pl.DataFrame:
        payload = self.client.get_json(
            query_url(
                self.endpoint,
                {"query": query, "mode": mode, "format": format, "maxrecords": maxrecords},
            )
        )
        rows = payload.get("timeline") if isinstance(payload, dict) else None
        if not rows:
            return pl.DataFrame()
        points = rows[0].get("data", rows) if isinstance(rows[0], dict) else rows
        if not isinstance(points, list):
            raise SourceError("GDELT timeline payload is not a list")
        flattened = []
        for point in points:
            flattened.append(
                {
                    "security_id": query,
                    "event_time": point.get("date"),
                    "available_time": utc_now(),
                    "value": point.get("value"),
                    "raw_value": json.dumps(point, sort_keys=True),
                }
            )
        return normalize_observations(flattened, source=self.name, value_key="value")


class SecEdgarSource(SourceAdapter):
    name = "sec_edgar"
    endpoint = "https://data.sec.gov/submissions/CIK{cik}.json"

    def fetch(self, *, cik: str) -> pl.DataFrame:
        normalized = str(cik).strip().upper().removeprefix("CIK")
        if not normalized.isdigit():
            raise ValueError("SEC CIK must contain only digits")
        payload = self.client.get_json(
            self.endpoint.format(cik=normalized.zfill(10)),
            headers={"Accept-Encoding": "gzip, deflate"},
        )
        recent = payload.get("filings", {}).get("recent", {}) if isinstance(payload, dict) else {}
        rows = [
            {
                "security_id": normalized,
                "event_time": filed,
                "available_time": utc_now(),
                "value": form,
                "title": accession,
            }
            for filed, form, accession in zip(
                recent.get("filingDate", []),
                recent.get("form", []),
                recent.get("accessionNumber", []),
                strict=False,
            )
        ]
        return (
            normalize_observations(rows, source=self.name, value_key="value")
            if rows
            else pl.DataFrame()
        )


class FredSource(SourceAdapter):
    name = "fred"
    endpoint = "https://api.stlouisfed.org/fred/series/observations"
    csv_endpoint = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def fetch(
        self,
        *,
        series_id: str,
        api_key: str | None = None,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> pl.DataFrame:
        if api_key or os.getenv("FRED_API_KEY"):
            payload = self.client.get_json(
                query_url(
                    self.endpoint,
                    {
                        "series_id": series_id,
                        "api_key": api_key or os.getenv("FRED_API_KEY"),
                        "file_type": "json",
                        "observation_start": observation_start,
                        "observation_end": observation_end,
                    },
                )
            )
            rows = [
                {
                    "security_id": series_id,
                    "event_time": item.get("date"),
                    "available_time": utc_now(),
                    "value": item.get("value"),
                }
                for item in payload.get("observations", [])
            ]
        else:
            text = self.client.get_text(
                query_url(
                    self.csv_endpoint,
                    {"id": series_id, "cosd": observation_start, "coed": observation_end},
                )
            )
            rows = [
                {
                    "security_id": series_id,
                    "event_time": row.get("observation_date"),
                    "available_time": utc_now(),
                    "value": row.get(series_id),
                }
                for row in csv_rows(text)
            ]
        return normalize_observations(rows, source=self.name)


class AlfredSource(FredSource):
    name = "alfred"
    csv_endpoint = "https://api.stlouisfed.org/fred/series/observations"


class TreasurySource(SourceAdapter):
    name = "us_treasury"
    endpoint = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"

    def fetch(
        self,
        *,
        endpoint: str = "v2/accounting/od/avg_interest_rates",
        filters: str | None = None,
        page_size: int = 1000,
    ) -> pl.DataFrame:
        payload = self.client.get_json(
            query_url(
                f"{self.endpoint}/{endpoint}",
                {"filter": filters, "page[size]": page_size, "format": "json"},
            )
        )
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        if not rows:
            return pl.DataFrame()
        date_key = next(
            (key for key in ("record_date", "effective_date", "date") if key in rows[0]), None
        )
        value_key = next(
            (key for key in ("avg_interest_rate_amt", "value", "total") if key in rows[0]), None
        )
        if not date_key or not value_key:
            raise SourceError("Treasury payload has no recognized date/value fields")
        return normalize_observations(
            [
                {
                    "security_id": str(row.get("security_type", endpoint)),
                    "event_time": row[date_key],
                    "available_time": utc_now(),
                    "value": row[value_key],
                }
                for row in rows
            ],
            source=self.name,
        )


class CftcSource(SourceAdapter):
    name = "cftc_cot"
    endpoint = "https://publicreporting.cftc.gov/resource/kh3c-gbw2.json"

    def fetch(self, *, limit: int = 5000, query: str | None = None) -> pl.DataFrame:
        payload = self.client.get_json(query_url(self.endpoint, {"$limit": limit, "$query": query}))
        if not isinstance(payload, list):
            raise SourceError("CFTC payload must be a list")
        date_key = next(
            (
                key
                for key in ("report_date_as_yyyymmdd", "report_date")
                if payload and key in payload[0]
            ),
            None,
        )
        value_key = next(
            (
                key
                for key in ("open_interest_all", "noncomm_positions_long_all")
                if payload and key in payload[0]
            ),
            None,
        )
        if not date_key or not value_key:
            return pl.DataFrame()
        return normalize_observations(
            [
                {
                    "security_id": row.get("market_and_exchange_names", self.name),
                    "event_time": row[date_key],
                    "available_time": utc_now(),
                    "value": row[value_key],
                }
                for row in payload
            ],
            source=self.name,
        )


class FinaSource(SourceAdapter):
    name = "finra_short_sale_volume"
    endpoint = "https://api.finra.org/data/group/otcMarket/name/regShoDaily"

    def fetch(self, *, url: str | None = None, limit: int = 1000) -> pl.DataFrame:
        payload = self.client.get_json(query_url(url or self.endpoint, {"limit": limit}))
        rows = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise SourceError("FINRA payload must be a list")
        date_key = next((key for key in ("tradeDate", "date") if rows and key in rows[0]), None)
        value_key = next(
            (key for key in ("shortVolume", "shortVolumeExempt") if rows and key in rows[0]), None
        )
        if not date_key or not value_key:
            return pl.DataFrame()
        return normalize_observations(
            [
                {
                    "security_id": row.get("symbol", self.name),
                    "event_time": row[date_key],
                    "available_time": utc_now(),
                    "value": row[value_key],
                }
                for row in rows
            ],
            source=self.name,
        )


class WorldBankSource(SourceAdapter):
    name = "world_bank"
    endpoint = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"

    def fetch(
        self, *, country: str = "US", indicator: str = "NY.GDP.MKTP.CD", page_size: int = 1000
    ) -> pl.DataFrame:
        payload = self.client.get_json(
            query_url(
                self.endpoint.format(country=country, indicator=indicator),
                {"format": "json", "per_page": page_size},
            )
        )
        if not isinstance(payload, list) or len(payload) < 2:
            raise SourceError("World Bank payload has unexpected shape")
        return normalize_observations(
            [
                {
                    "security_id": f"{country}:{indicator}",
                    "event_time": f"{row.get('date')}-12-31",
                    "available_time": utc_now(),
                    "value": row.get("value"),
                }
                for row in payload[1]
                if row.get("value") is not None
            ],
            source=self.name,
        )


class BeaSource(SourceAdapter):
    name = "bea"
    endpoint = "https://apps.bea.gov/api/data/"

    def fetch(
        self,
        *,
        dataset: str = "NIPA",
        table_name: str = "T10101",
        api_key: str | None = None,
        user_id: str | None = None,
    ) -> pl.DataFrame:
        key = api_key or os.getenv("BEA_API_KEY") or user_id
        if not key:
            raise SourceError("BEA requires api_key or BEA_API_KEY; credentials are never embedded")
        payload = self.client.get_json(
            query_url(
                self.endpoint,
                {
                    "UserID": key,
                    "method": "GETDATA",
                    "datasetname": dataset,
                    "TableName": table_name,
                    "ResultFormat": "JSON",
                },
            )
        )
        data = (
            payload.get("BEAAPI", {}).get("Results", {}).get("Data", [])
            if isinstance(payload, dict)
            else []
        )
        rows = [
            {
                "security_id": table_name,
                "event_time": row.get("TimePeriod"),
                "value": row.get("DataValue"),
            }
            for row in data
            if row.get("TimePeriod") and row.get("DataValue") not in (None, "")
        ]
        return normalize_observations(rows, source=self.name) if rows else pl.DataFrame()


class ItchSampleSource(SourceAdapter):
    name = "nasdaq_itch"

    def fetch(self, *, path: str | Path) -> pl.DataFrame:
        text = Path(path).read_text(encoding="utf-8")
        rows = csv_rows(text)
        return normalize_ohlcv(rows, source=self.name, symbol_key="security_id")


class Fi2010Source(SourceAdapter):
    name = "fi_2010"

    def fetch(self, *, path: str | Path) -> pl.DataFrame:
        frame = pl.read_csv(Path(path), separator=",", try_parse_dates=True)
        if frame.is_empty():
            return frame
        if "event_time" not in frame.columns:
            raise SourceError("FI-2010 file must include event_time")
        return frame.with_columns(pl.lit(self.name).alias("source"))


class _OptionalLibrarySource(SourceAdapter):
    library_name = "optional"

    def fetch(self, *, payload: Any = None, **kwargs: Any) -> pl.DataFrame:
        if payload is not None:
            if isinstance(payload, pl.DataFrame):
                return payload
            if isinstance(payload, list):
                return pl.DataFrame(payload)
        raise SourceError(
            f"{self.name} requires an explicit {self.library_name} payload; no dependency is installed by default"
        )


class CryptofeedSource(_OptionalLibrarySource):
    name = "cryptofeed"
    library_name = "cryptofeed"


class CcxtSource(_OptionalLibrarySource):
    name = "ccxt"
    library_name = "ccxt"


class OpenBBSource(_OptionalLibrarySource):
    name = "openbb"
    library_name = "openbb"
