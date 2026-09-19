"""Yahoo Finance v8 daily chart as a PIT-shaped file tape. Not SIP vintages.

Yahoo EOD OHLC is treated as vendor-adjusted (``revision_id=YAHOO_VENDOR_ADJ``).
``available_time`` equals session close. Same honesty contract as the Stooq
writer: scientific scoring only, no champion alias, no live claim.

Stooq EOD is preferred when it returns CSV; Yahoo is the fallback when Stooq
serves a JavaScript proof-of-work wall.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from quant_fund.data.adapters.stooq import (
    US_LIQUID,
    USER_AGENT,
    np_finite,
    session_close,
    write_file_lake,
)

SOURCE = "yahoo"
REVISION = "YAHOO_VENDOR_ADJ"
CHART_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    "?period1={start}&period2={end}&interval=1d&includeAdjustedClose=true"
)

YAHOO_US: tuple[tuple[str, str], ...] = tuple((sid, sid) for sid, _stooq in US_LIQUID)
YAHOO_UK: tuple[tuple[str, str], ...] = (
    ("VOD", "VOD.L"),
    ("BP", "BP.L"),
    ("HSBA", "HSBA.L"),
    ("AZN", "AZN.L"),
    ("ULVR", "ULVR.L"),
    ("SHEL", "SHEL.L"),
)


def fetch_yahoo_chart(
    symbol: str,
    *,
    start: datetime,
    end: datetime,
    timeout: float = 30.0,
) -> dict:
    url = CHART_URL.format(
        symbol=symbol,
        start=int(start.timestamp()),
        end=int(end.timestamp()),
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — public chart JSON
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("yahoo chart payload is not an object")
    return payload


def parse_yahoo_chart(payload: dict, *, security_id: str, yahoo_symbol: str) -> pl.DataFrame:
    chart = payload.get("chart") if isinstance(payload, dict) else None
    if not isinstance(chart, dict):
        return pl.DataFrame()
    result = chart.get("result")
    if not isinstance(result, list) or not result:
        return pl.DataFrame()
    block = result[0]
    if not isinstance(block, dict):
        return pl.DataFrame()
    stamps = block.get("timestamp") or []
    indicators = block.get("indicators") or {}
    quotes = (indicators.get("quote") or [{}])[0]
    opens = quotes.get("open") or []
    highs = quotes.get("high") or []
    lows = quotes.get("low") or []
    closes = quotes.get("close") or []
    volumes = quotes.get("volume") or []
    ingested = datetime.now(tz=UTC)
    suffix = ".uk" if yahoo_symbol.endswith(".L") else ".us"
    rows: list[dict[str, object]] = []
    n = min(len(stamps), len(opens), len(highs), len(lows), len(closes), len(volumes))
    for i in range(n):
        try:
            opn = float(opens[i])
            high = float(highs[i])
            low = float(lows[i])
            close = float(closes[i])
            volume = float(volumes[i] or 0.0)
        except (TypeError, ValueError):
            continue
        if not all(np_finite(v) and v > 0 for v in (opn, high, low, close)):
            continue
        if volume < 0 or not np_finite(volume):
            continue
        lo = min(opn, high, low, close)
        hi = max(opn, high, low, close)
        day = datetime.fromtimestamp(int(stamps[i]), tz=UTC).date()
        ts = session_close(day, suffix)
        rows.append(
            {
                "security_id": security_id,
                "symbol": security_id,
                "event_time": ts,
                "available_time": ts,
                "ingested_time": ingested,
                "source": SOURCE,
                "revision_id": REVISION,
                "open": float(opn),
                "high": float(hi),
                "low": float(lo),
                "close": float(close),
                "volume": float(volume),
                "currency": "GBP" if suffix == ".uk" else "USD",
                "session": "rth",
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def download_yahoo_universe(
    root: Path,
    names: tuple[tuple[str, str], ...] = YAHOO_US,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    pause_s: float = 0.15,
) -> dict[str, object]:
    start_ts = start or datetime(2019, 1, 2, tzinfo=UTC)
    end_ts = end or datetime.now(tz=UTC)
    frames: list[pl.DataFrame] = []
    errors: dict[str, str] = {}
    for security_id, symbol in names:
        try:
            payload = fetch_yahoo_chart(symbol, start=start_ts, end=end_ts)
            frame = parse_yahoo_chart(payload, security_id=security_id, yahoo_symbol=symbol)
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            errors[security_id] = str(exc)
            time.sleep(pause_s)
            continue
        if not frame.is_empty():
            frames.append(frame)
        time.sleep(pause_s)
    if not frames:
        return {
            "status": "empty",
            "n_names": 0,
            "errors": errors,
            "source": SOURCE,
            "pit_convention": "session_close_equals_available",
            "vendor_adjusted": True,
            "sip_vintage": False,
        }
    bars = pl.concat(frames, how="diagonal_relaxed").sort(["event_time", "security_id"])
    paths = write_file_lake(bars, root)
    return {
        "status": "ok",
        "n_names": int(bars["security_id"].n_unique()),
        "n_bars": int(bars.height),
        "paths": {k: str(v) for k, v in paths.items()},
        "errors": errors,
        "source": SOURCE,
        "revision_id": REVISION,
        "pit_convention": "session_close_equals_available",
        "vendor_adjusted": True,
        "sip_vintage": False,
        "champion_alias": False,
        "research_only": True,
    }
