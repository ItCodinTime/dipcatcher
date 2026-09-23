"""Fetch OKX delta-neutral carry data: perp bars + spot bars + funding events.

Universe: top-N live USDT-margined SWAPs by 24h dollar volume that also have a
live USDT spot pair. Writes three parquets shaped for ``run_carry_backtest``:

- data/okx_carry/perp_bars.parquet  (event_time, security_id, ohlcv)
- data/okx_carry/spot_bars.parquet  (same)
- data/okx_carry/funding.parquet    (event_time, security_id, value=rate)

security_id is the base coin (BTC) so perp BTC-USDT-SWAP pairs with BTC-USDT.
Research data collection only — public endpoints, no auth, no trading claim.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

BASE = "https://www.okx.com"


def _get(path: str, *, retries: int = 6) -> dict:
    url = BASE + path
    delay = 0.4
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "dipcatcher-research/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                payload = json.load(r)
            if payload.get("code") not in ("0", 0):
                raise RuntimeError(f"okx error {payload.get('code')}: {payload.get('msg')}")
            return payload
        except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError):
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 8.0)
    raise RuntimeError("unreachable")


def _get_throttled(path: str, spacing_s: float = 0.12) -> dict:
    time.sleep(spacing_s)
    return _get(path)


def list_universe(top_n: int) -> list[str]:
    inst = _get("/api/v5/public/instruments?instType=SWAP")["data"]
    swaps = {
        x["instId"] for x in inst if x["instId"].endswith("-USDT-SWAP") and x.get("state") == "live"
    }
    tick = _get("/api/v5/market/tickers?instType=SWAP")["data"]
    dv: dict[str, float] = {}
    for t in tick:
        iid = t["instId"]
        if iid not in swaps:
            continue
        try:
            dv[iid] = float(t["volCcy24h"]) * float(t["last"])
        except (KeyError, ValueError, TypeError):
            continue
    spot_inst = _get("/api/v5/public/instruments?instType=SPOT")["data"]
    spot_ids = {x["instId"] for x in spot_inst if x.get("state") == "live"}
    ranked = sorted(dv.items(), key=lambda kv: kv[1], reverse=True)
    out: list[str] = []
    for iid, _vol in ranked:
        base = iid.split("-")[0]
        if f"{base}-USDT" in spot_ids:
            out.append(base)
        if len(out) >= top_n:
            break
    return out


def fetch_candles(inst_id: str, bar: str = "1D", limit: int = 100) -> list[list[str]]:
    """All daily history for inst_id; OKX paginates older via ``after=<ts>``."""
    rows: list[list[str]] = []
    cursor: str | None = None
    seen: set[str] = set()
    while True:
        q = f"/api/v5/market/history-candles?instId={inst_id}&bar={bar}&limit={limit}"
        if cursor is not None:
            q += f"&after={cursor}"
        data = _get_throttled(q)["data"]
        if not data:
            break
        rows.extend(data)
        oldest = data[-1][0]
        if oldest in seen or len(data) < limit:
            break
        seen.add(oldest)
        cursor = oldest
        if len(rows) > 6000:
            break
    return rows


def fetch_funding(inst_id: str, limit: int = 100) -> list[dict]:
    rows: list[dict] = []
    cursor: str | None = None
    seen: set[str] = set()
    while True:
        q = f"/api/v5/public/funding-rate-history?instId={inst_id}&limit={limit}"
        if cursor is not None:
            q += f"&after={cursor}"
        data = _get_throttled(q)["data"]
        if not data:
            break
        rows.extend(data)
        oldest = data[-1]["fundingTime"]
        if oldest in seen or len(data) < limit:
            break
        seen.add(oldest)
        cursor = oldest
        if len(rows) > 20000:
            break
    return rows


def _bars_frame(coin: str, candles: list[list[str]], source: str) -> pl.DataFrame:
    recs = []
    for c in candles:
        # [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
        ts = datetime.fromtimestamp(int(c[0]) / 1000, tz=UTC)
        recs.append(
            {
                "event_time": ts,
                "security_id": coin,
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[7]) if len(c) > 7 and c[7] else float(c[5]),
                "source": source,
            }
        )
    return pl.DataFrame(recs).unique(["event_time", "security_id"]).sort("event_time")


def _fund_frame(coin: str, events: list[dict]) -> pl.DataFrame:
    recs = [
        {
            "event_time": datetime.fromtimestamp(int(e["fundingTime"]) / 1000, tz=UTC),
            "security_id": coin,
            "value": float(e["fundingRate"]),
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


def _one(coin: str) -> tuple[str, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    perp = _bars_frame(coin, fetch_candles(f"{coin}-USDT-SWAP"), "okx")
    spot = _bars_frame(coin, fetch_candles(f"{coin}-USDT"), "okx")
    fund = _fund_frame(coin, fetch_funding(f"{coin}-USDT-SWAP"))
    return coin, perp, spot, fund


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--out", type=Path, default=Path("data/okx_carry"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    coins = list_universe(args.top)
    print(f"universe: {len(coins)} coins: {coins[:12]}...")
    perps, spots, funds = [], [], []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, c): c for c in coins}
        for fut in cf.as_completed(futs):
            coin = futs[fut]
            try:
                _, p, s, f = fut.result()
                if p.height and s.height:
                    perps.append(p)
                    spots.append(s)
                    if f.height:
                        funds.append(f)
                print(f"  {coin}: perp={p.height} spot={s.height} fund={f.height}")
            except Exception as e:  # noqa: BLE001
                print(f"  {coin}: FAILED {e}")
    args.out.mkdir(parents=True, exist_ok=True)
    perp_df = pl.concat(perps) if perps else pl.DataFrame()
    spot_df = pl.concat(spots) if spots else pl.DataFrame()
    fund_df = pl.concat(funds) if funds else pl.DataFrame()
    perp_df.write_parquet(args.out / "perp_bars.parquet")
    spot_df.write_parquet(args.out / "spot_bars.parquet")
    fund_df.write_parquet(args.out / "funding.parquet")
    print(
        f"wrote {args.out}: perp={perp_df.height} spot={spot_df.height} funding={fund_df.height} "
        f"coins={len(perps)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
