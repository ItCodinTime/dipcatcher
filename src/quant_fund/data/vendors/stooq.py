"""Stooq daily CSV adapter — keyless public file endpoint.

Wire contract: ``GET https://stooq.com/q/d/l/?s={symbol}&i=d&d1=&d2=`` returns
``Date,Open,High,Low,Close,Volume`` CSV (or a JS challenge / ``Exceeded the
daily hits limit`` text on throttle). Symbols map like the repo's file-tape
adapter: ``SPY`` → ``spy.us``; an explicit ``.us``/``.uk`` suffix is honored,
``*.uk`` stamps the 16:30 London close.

Same conventions as ``quant_fund.data.adapters.stooq``: bar close =
``event_time == available_time``, envelope repair ``lo=min(O,H,L,C)`` /
``hi=max(O,H,L,C)``, ``STOOQ_VENDOR_ADJ`` revision. No published rate limit;
the default pace (1 req/s) is a politeness bound, not a documented cap.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from quant_fund.data.sources.normalize import csv_rows
from quant_fund.data.vendors.base import DailyBarsAdapter, vendor_session_close
from quant_fund.data.vendors.errors import VendorResponseError

_CSV_FIELDS = ("Date", "Open", "High", "Low", "Close", "Volume")


def stooq_symbol_for(symbol: str) -> str:
    """``SPY`` → ``spy.us``; explicit suffixes pass through lowercased."""
    cleaned = symbol.strip().lower()
    if not cleaned:
        raise ValueError("symbol must be non-empty")
    return cleaned if "." in cleaned else f"{cleaned}.us"


class StooqDailyAdapter(DailyBarsAdapter):
    name = "stooq"
    env_key = None  # keyless public CSV endpoint
    base_url = "https://stooq.com"
    revision_id = "STOOQ_VENDOR_ADJ"
    response_kind = "text"
    default_rate_per_second = 1.0
    default_capacity = 2.0

    def build_request(
        self,
        symbol: str,
        start: datetime | None,
        end: datetime | None,
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        stooq_symbol = stooq_symbol_for(symbol)
        params: dict[str, Any] = {"s": stooq_symbol, "i": "d"}
        if start is not None:
            params["d1"] = start.strftime("%Y%m%d")
        if end is not None:
            params["d2"] = end.strftime("%Y%m%d")
        return "/q/d/l/", params, {}

    def rows_from_payload(self, payload: Any, symbol: str) -> list[dict[str, Any]]:
        if not isinstance(payload, str):
            raise VendorResponseError(
                f"stooq: expected CSV text, got {type(payload).__name__}"
            )
        # Throttle/challenge pages arrive as HTML or a bare text notice.
        if "<html" in payload.lower() or "hits limit" in payload.lower():
            raise VendorResponseError("stooq: throttle/challenge page, not CSV")
        rows = csv_rows(payload)
        stooq_symbol = stooq_symbol_for(symbol)
        out: list[dict[str, Any]] = []
        for raw in rows:
            missing = [name for name in _CSV_FIELDS if (raw.get(name) or "").strip() == ""]
            if "Date" in missing:
                continue  # skip blank separator lines
            if missing:
                raise VendorResponseError(
                    f"stooq: CSV row missing fields {missing}: {raw!r}"
                )
            try:
                day = datetime.strptime(raw["Date"].strip(), "%Y-%m-%d").date()
                opn = float(raw["Open"])
                high = float(raw["High"])
                low = float(raw["Low"])
                close = float(raw["Close"])
                volume = float(raw["Volume"])
            except (TypeError, ValueError) as exc:
                raise VendorResponseError(
                    f"stooq: unparseable CSV row: {raw!r}"
                ) from exc
            close_ts = vendor_session_close(day, stooq_symbol)
            rows_min, rows_max = min(opn, high, low, close), max(opn, high, low, close)
            out.append(
                {
                    "security_id": symbol.strip().upper(),
                    "event_time": close_ts,
                    "available_time": close_ts,
                    # Envelope repair identical to the file-tape adapter.
                    "open": opn,
                    "high": rows_max,
                    "low": rows_min,
                    "close": close,
                    "volume": volume,
                }
            )
        return out
