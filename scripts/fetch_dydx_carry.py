"""Fetch dYdX v4 delta-neutral carry data: perp bars + funding events.

Universe: every ACTIVE perpetualMarket on the dYdX v4 indexer (~180 tickers,
history back to chain genesis Oct 2023). Writes two parquets into
``data/dydx_carry``:

- perp_bars.parquet  (event_time, security_id, ohlcv) — security_id = base coin
- funding.parquet    (event_time, security_id, value=rate) — hourly events

dYdX has no spot market, so the hedge leg always resolves to Binance spot in
build_dydx_book.py, which also prefixes ids ``DYDX:<coin>`` for collision-free
merge with the Binance carry universe via ``--extra-dir``.

Research data collection only — public indexer endpoints, no auth, no trading.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

BASE = "https://indexer.dydx.trade"


_LOCK = threading.Lock()
_LAST_CALL = [0.0]
_SPACING = 0.5  # serialized calls/sec across workers; indexer rate-limits bursts


def _get(path: str, *, retries: int = 14) -> dict:
    delay = 0.4
    for attempt in range(retries):
        with _LOCK:
            wait = _SPACING - (time.monotonic() - _LAST_CALL[0])
            if wait > 0:
                time.sleep(wait)
            _LAST_CALL[0] = time.monotonic()
        try:
            req = urllib.request.Request(
                BASE + path,
                headers={"User-Agent": "dipcatcher-research/1.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500 and e.code != 429:
                raise
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 1.5, 15.0)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 8.0)
    raise RuntimeError("unreachable")


def list_universe() -> list[str]:
    markets = _get("/v4/perpetualMarkets")["markets"]
    return sorted(
        t.split("-")[0]
        for t, m in markets.items()
        if m.get("status") == "ACTIVE" and t.endswith("-USD")
    )


def fetch_candles(ticker: str, resolution: str = "1DAY", limit: int = 100) -> list[dict]:
    """All daily candles for ticker; indexer paginates older via toISO."""
    rows: list[dict] = []
    to_iso = "2027-01-01T00:00:00.000Z"
    while True:
        data = _get(
            f"/v4/candles/perpetualMarkets/{ticker}"
            f"?resolution={resolution}&limit={limit}&toISO={to_iso}"
        ).get("candles", [])
        if not data:
            break
        rows.extend(data)
        oldest = min(c["startedAt"] for c in data)
        if oldest == to_iso or len(data) < limit:
            break
        to_iso = oldest
        if len(rows) > 20000:
            break
    return rows


def fetch_funding(ticker: str, limit: int = 1000) -> list[dict]:
    """All hourly funding events; paginate older via effectiveBeforeOrAt."""
    rows: list[dict] = []
    before = "2027-01-01T00:00:00.000Z"
    while True:
        data = _get(
            f"/v4/historicalFunding/{ticker}?limit={limit}&effectiveBeforeOrAt={before}"
        ).get("historicalFunding", [])
        if not data:
            break
        rows.extend(data)
        oldest = min(e["effectiveAt"] for e in data)
        if oldest >= before or len(data) < limit:
            break
        before = oldest
        if len(rows) > 500000:
            break
    return rows


def _bars_frame(coin: str, candles: list[dict], source: str) -> pl.DataFrame:
    recs = [
        {
            "event_time": datetime.fromisoformat(c["startedAt"].replace("Z", "+00:00")).astimezone(
                UTC
            ),
            "security_id": coin,
            "open": float(c["open"]),
            "high": float(c["high"]),
            "low": float(c["low"]),
            "close": float(c["close"]),
            "volume": float(c["baseTokenVolume"]),
            "source": source,
        }
        for c in candles
    ]
    return (
        pl.DataFrame(recs).unique(["event_time", "security_id"]).sort("event_time")
        if recs
        else pl.DataFrame(
            schema={
                "event_time": pl.Datetime("us", UTC),
                "security_id": pl.String,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "source": pl.String,
            }
        )
    )


def _fund_frame(coin: str, events: list[dict]) -> pl.DataFrame:
    recs = [
        {
            "event_time": datetime.fromisoformat(
                e["effectiveAt"].replace("Z", "+00:00")
            ).astimezone(UTC),
            "security_id": coin,
            "value": float(e["rate"]),
        }
        for e in events
    ]
    return (
        pl.DataFrame(recs).unique(["event_time", "security_id"]).sort("event_time")
        if recs
        else pl.DataFrame(
            schema={
                "event_time": pl.Datetime("us", UTC),
                "security_id": pl.String,
                "value": pl.Float64,
            }
        )
    )


def _one(ticker: str) -> tuple[str, pl.DataFrame, pl.DataFrame]:
    coin = ticker.split("-")[0]
    perp = _bars_frame(coin, fetch_candles(ticker), "dydx")
    fund = _fund_frame(coin, fetch_funding(ticker))
    return coin, perp, fund


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/dydx_carry"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--coins", type=str, default="", help="comma list of base coins")
    args = ap.parse_args()

    coins = args.coins.split(",") if args.coins else list_universe()
    print(f"universe: {len(coins)} markets", flush=True)

    perps, funds = [], []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, f"{c}-USD"): c for c in coins}
        for fut in cf.as_completed(futs):
            coin = futs[fut]
            try:
                _, p, f = fut.result()
                if p.height:
                    perps.append(p)
                if f.height:
                    funds.append(f)
                print(f"  {coin}: perp={p.height} fund={f.height}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  {coin}: FAILED {e}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    perp_df = pl.concat(perps) if perps else pl.DataFrame()
    fund_df = pl.concat(funds) if funds else pl.DataFrame()
    perp_df.write_parquet(args.out / "perp_bars.parquet")
    fund_df.write_parquet(args.out / "funding.parquet")
    print(f"wrote {args.out}: perp={perp_df.height} funding={fund_df.height} coins={len(perps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
