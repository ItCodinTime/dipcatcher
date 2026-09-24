"""Fetch Hyperliquid delta-neutral carry data: perp bars + funding events + HL spot bars.

Universe: every non-delisted perp in the ``meta`` universe (~230 names, history
back to each coin's listing from 2023+). Writes three parquets shaped for
``run_carry_backtest`` into ``data/hyperliquid_carry``:

- perp_bars.parquet      (event_time, security_id, ohlcv) — security_id = coin
- hl_spot_bars.parquet   (same) — HL-native spot pairs whose base token matches a perp
- funding.parquet        (event_time, security_id, value=rate) — hourly events

A separate build step (scripts/build_hl_book.py) resolves the spot hedge leg
(HL spot if the token has an HL spot pair, else Binance spot) and prefixes
ids to ``HL:<coin>`` so the book can merge with the Binance carry universe
without collisions.

Research data collection only — public ``/info`` endpoint, no auth, no trading.
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

INFO = "https://api.hyperliquid.xyz/info"
DAY_MS = 86400000
MS = 1000


_LOCK = threading.Lock()
_LAST_CALL = [0.0]
_SPACING = 0.55  # serialized calls/sec across all workers; HL /info weight-limits bursts


def _post(body: dict, *, retries: int = 14) -> object:
    delay = 0.5
    for attempt in range(retries):
        with _LOCK:
            wait = _SPACING - (time.monotonic() - _LAST_CALL[0])
            if wait > 0:
                time.sleep(wait)
            _LAST_CALL[0] = time.monotonic()
        try:
            req = urllib.request.Request(
                INFO,
                data=json.dumps(body).encode(),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "dipcatcher-research/1.0",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            # 429 rate limit: always retry with backoff; other 4xx are data errors
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
            delay = min(delay * 2, 10.0)
    raise RuntimeError("unreachable")


def list_perp_universe() -> list[str]:
    meta = _post({"type": "meta"})
    assert isinstance(meta, dict)
    out = []
    for u in meta.get("universe", []):
        if u.get("isDelisted"):
            continue
        name = u["name"]
        # builder-deployed perps are namespaced "xyz:ABC"; keep native markets
        if ":" in name:
            continue
        out.append(name)
    return out


def list_spot_pairs() -> dict[str, str]:
    """Map base token name -> spot universe entry name for candleSnapshot."""
    meta = _post({"type": "spotMeta"})
    assert isinstance(meta, dict)
    idx_to_token = {t["index"]: t["name"] for t in meta.get("tokens", [])}
    out: dict[str, str] = {}
    for u in meta.get("universe", []):
        toks = u.get("tokens", [])
        if len(toks) != 2:
            continue
        base = idx_to_token.get(toks[0])
        quote = idx_to_token.get(toks[1])
        if base and quote == "USDC":
            out[base] = u["name"]
    return out


def fetch_candles(coin: str, interval: str = "1d") -> list[dict]:
    """All candles for coin (perp name or spot universe entry name)."""
    rows: list[dict] = []
    end = int(datetime.now(tz=UTC).timestamp() * MS)
    start = 1672531200000  # HL mainnet genesis era; snapshot returns from listing
    while True:
        data = _post(
            {
                "type": "candleSnapshot",
                "req": {"coin": coin, "interval": interval, "startTime": start, "endTime": end},
            }
        )
        assert isinstance(data, list)
        if not data:
            break
        rows.extend(data)
        newest_t = max(int(c["t"]) for c in data)
        if newest_t < start or len(data) < 2:
            break
        if newest_t >= end - DAY_MS:
            break
        start = newest_t + 1
        if len(rows) > 30000:
            break
    return rows


def sample_funding(coin: str, start: int, end: int) -> float:
    """Sum of rates over the first 500 events of [start, end) — prescreen probe."""
    data = _post({"type": "fundingHistory", "coin": coin, "startTime": start, "endTime": end})
    assert isinstance(data, list)
    return sum(float(e["fundingRate"]) for e in data)


def prescreen(coins: list[str], thresh: float = 0.0008) -> list[str]:
    """Keep coins whose funding was material in at least one life-stage window.

    Three 500-event samples per coin (listing, midpoint, recent) — coins flat
    in all three cannot sustain book membership (the entry bar needs ~2bp/day
    for 9+ days), so skipping their full history loses no held position.
    """
    now = int(datetime.now(tz=UTC).timestamp() * MS)
    genesis = 1672531200000
    mid = (genesis + now) // 2
    windows = [(genesis, mid), (mid, now - 21 * DAY_MS), (now - 21 * DAY_MS, now)]
    keep: list[str] = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {
            ex.submit(sample_funding, c, s, e): (c, tag)
            for c in coins
            for (s, e), tag in zip(windows, ("early", "mid", "late"), strict=True)
        }
        scores: dict[str, list[float]] = {c: [] for c in coins}
        for fut in cf.as_completed(futs):
            c, tag = futs[fut]
            try:
                scores[c].append(abs(fut.result()))
            except Exception as e:  # noqa: BLE001
                print(f"  {c} {tag}: FAILED {e}", flush=True)
    for c in coins:
        if scores[c] and max(scores[c]) >= thresh:
            keep.append(c)
    return keep


def fetch_funding(coin: str) -> list[dict]:
    """All hourly funding events; API returns max 500, forward-paginated."""
    rows: list[dict] = []
    end = int(datetime.now(tz=UTC).timestamp() * MS)
    start = 1672531200000
    while True:
        data = _post({"type": "fundingHistory", "coin": coin, "startTime": start, "endTime": end})
        assert isinstance(data, list)
        if not data:
            break
        rows.extend(data)
        if len(data) < 500:
            break
        time.sleep(0.15)
        newest = max(int(e["time"]) for e in data)
        if newest <= start:
            break
        start = newest + 1
        if len(rows) > 500000:
            break
    return rows


def _bars_frame(coin: str, candles: list[dict], source: str) -> pl.DataFrame:
    recs = []
    for c in candles:
        recs.append(
            {
                "event_time": datetime.fromtimestamp(int(c["t"]) / MS, tz=UTC),
                "security_id": coin,
                "open": float(c["o"]),
                "high": float(c["h"]),
                "low": float(c["l"]),
                "close": float(c["c"]),
                "volume": float(c["v"]),
                "source": source,
            }
        )
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
            "event_time": datetime.fromtimestamp(int(e["time"]) / MS, tz=UTC),
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


def _one(
    coin: str,
    spot_name: str | None,
    interval: str = "1d",
    want_spot: bool = True,
    want_funding: bool = True,
) -> tuple[str, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    perp = _bars_frame(coin, fetch_candles(coin, interval), "hyperliquid")
    spot = (
        _bars_frame(coin, fetch_candles(spot_name), "hyperliquid")
        if spot_name and want_spot
        else _bars_frame(coin, [], "hyperliquid")
    )
    fund = _fund_frame(coin, fetch_funding(coin)) if want_funding else _fund_frame(coin, [])
    return coin, perp, spot, fund


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/hyperliquid_carry"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--coins", type=str, default="", help="comma list override")
    ap.add_argument("--interval", default="1d", help="candle interval for perp bars")
    ap.add_argument("--skip-spot", action="store_true")
    ap.add_argument("--skip-funding", action="store_true")
    ap.add_argument(
        "--prescreen",
        action="store_true",
        help="only deep-fetch coins with material funding in any life-stage window",
    )
    args = ap.parse_args()

    coins = args.coins.split(",") if args.coins else list_perp_universe()
    if args.prescreen:
        coins = prescreen(coins)
        print(f"prescreen kept {len(coins)} coins with material funding", flush=True)
    spot_map = list_spot_pairs()
    hedgeable = [c for c in coins if c in spot_map]
    print(
        f"universe: {len(coins)} perps, {len(hedgeable)} with HL spot leg; "
        f"the rest hedge via Binance spot in build_hl_book.py"
    )

    perps, spots, funds = [], [], []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                _one, c, spot_map.get(c), args.interval, not args.skip_spot, not args.skip_funding
            ): c
            for c in coins
        }
        for fut in cf.as_completed(futs):
            coin = futs[fut]
            try:
                _, p, s, f = fut.result()
                if p.height:
                    perps.append(p)
                if s.height:
                    spots.append(s)
                if f.height:
                    funds.append(f)
                print(
                    f"  {coin}: perp={p.height} spot={s.height} fund={f.height}",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  {coin}: FAILED {e}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    perp_df = pl.concat(perps) if perps else pl.DataFrame()
    spot_df = pl.concat(spots) if spots else pl.DataFrame()
    fund_df = pl.concat(funds) if funds else pl.DataFrame()
    perp_df.write_parquet(args.out / "perp_bars.parquet")
    if not args.skip_spot:
        spot_df.write_parquet(args.out / "hl_spot_bars.parquet")
    if not args.skip_funding:
        fund_df.write_parquet(args.out / "funding.parquet")
    print(
        f"wrote {args.out}: perp={perp_df.height} hl_spot={spot_df.height} "
        f"funding={fund_df.height} coins={len(perps)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
