"""Top-up fetch: Binance daily-granularity klines for dates the monthly
archive doesn't cover (spot 2025+; perp past the last monthly file).

Merges into data/binance_carry/{perp_bars,spot_bars}.parquet (dedup on
event_time+security_id). Research data collection only.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import time
import urllib.error
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
from fetch_binance_vision_carry import CANDIDATES, S3, _dl_zip_csv, _parse_kline_csv, list_keys

DATA = Path("data/binance_carry")
TODAY = datetime.now(UTC).date()


def _days_for(coin: str, kind: str, start: date) -> list[date]:
    """List available daily zip dates for {kind}/{coin}USDT/1d from start.

    Pages the S3 listing starting at the marker just before ``start`` so the
    pre-start archive is skipped instead of paginated through.
    """
    sym = f"{coin}USDT"
    prefix = f"data/{kind}/daily/klines/{sym}/1d/{sym}-1d-"
    keys = list_keys(prefix, marker=f"{prefix}{start.isoformat()}")
    out = []
    for k in keys:
        try:
            d = date.fromisoformat(k.removesuffix(".zip")[-10:])
        except ValueError:
            continue
        if start <= d <= TODAY:
            out.append(d)
    return sorted(set(out))


def fetch_coin_tail(coin: str, kind: str, start: date) -> list[dict]:
    sym = f"{coin}USDT"
    rows: list[dict] = []
    for d in _days_for(coin, kind, start):
        url = f"{S3}/data/{kind}/daily/klines/{sym}/1d/{sym}-1d-{d.isoformat()}.zip"
        raw = _dl_zip_csv(url)
        if raw:
            rows.extend(_parse_kline_csv(raw, coin, "binance_daily"))
    return rows


def _fapi_funding(coin: str, start_ms: int, end_ms: int) -> list[dict]:
    """Public REST funding history; monthly Vision archives don't cover the tail."""
    rows: list[dict] = []
    cursor = start_ms
    while True:
        url = (
            f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={coin}USDT"
            f"&startTime={cursor}&endTime={end_ms}&limit=1000"
        )
        for attempt in range(6):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "dipcatcher-research/1.0"})
                with urllib.request.urlopen(req, timeout=20) as r:
                    data = json.load(r)
                break
            except (urllib.error.URLError, TimeoutError):
                if attempt == 5:
                    raise
                time.sleep(0.5 * (attempt + 1))
        if not data:
            break
        rows.extend(data)
        if len(data) < 1000:
            break
        nxt = max(int(e["fundingTime"]) for e in data) + 1
        if nxt <= cursor:
            break
        cursor = nxt
    return rows


def merge_funding_tail(data_dir: Path, start: date, coins: list[str], workers: int) -> None:
    path = data_dir / "funding.parquet"
    existing = pl.read_parquet(path)
    start_ms = int(datetime(start.year, start.month, start.day, tzinfo=UTC).timestamp() * 1000)
    end_ms = int(TODAY.toordinal() * 86400) * 1000
    have = set(
        existing.filter(
            pl.col("event_time") >= datetime(start.year, start.month, start.day, tzinfo=UTC)
        )
        .select("event_time", "security_id")
        .iter_rows()
    )
    all_rows: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fapi_funding, c, start_ms, end_ms): c for c in coins}
        for fut in cf.as_completed(futs):
            c = futs[fut]
            try:
                rows = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  {c} funding: FAILED {e}", flush=True)
                continue
            recs = [
                {
                    "event_time": datetime.fromtimestamp(int(e["fundingTime"]) / 1000, tz=UTC),
                    "security_id": c,
                    "value": float(e["fundingRate"]),
                }
                for e in rows
            ]
            recs = [r for r in recs if (r["event_time"], r["security_id"]) not in have]
            all_rows.extend(recs)
            print(f"  {c} funding: +{len(recs)}", flush=True)
    if not all_rows:
        print("funding: nothing new")
        return
    add = pl.DataFrame(all_rows, schema=existing.schema)
    merged = (
        pl.concat([existing, add])
        .unique(["event_time", "security_id"], keep="first")
        .sort(["security_id", "event_time"])
    )
    merged.write_parquet(path)
    print(f"funding: merged +{add.height} -> {merged.height} rows at {path}")


def merge_tail(data_dir: Path, kind: str, start: date, coins: list[str], workers: int) -> None:
    path = data_dir / ("perp_bars.parquet" if kind == "futures/um" else "spot_bars.parquet")
    existing = pl.read_parquet(path)
    have = set(
        existing.filter(
            pl.col("event_time") >= datetime(start.year, start.month, start.day, tzinfo=UTC)
        )
        .select("event_time", "security_id")
        .iter_rows()
    )
    all_rows: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_coin_tail, c, kind, start): c for c in coins}
        for fut in cf.as_completed(futs):
            c = futs[fut]
            try:
                rows = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  {c} {kind}: FAILED {e}", flush=True)
                continue
            rows = [r for r in rows if (r["event_time"], r["security_id"]) not in have]
            all_rows.extend(rows)
            print(f"  {c} {kind}: +{len(rows)} bars", flush=True)
    if not all_rows:
        print(f"{kind}: nothing new")
        return
    add = pl.DataFrame(all_rows, schema=existing.schema)
    merged = (
        pl.concat([existing, add])
        .unique(["event_time", "security_id"], keep="first")
        .sort(["security_id", "event_time"])
    )
    merged.write_parquet(path)
    print(f"{kind}: merged +{add.height} -> {merged.height} rows at {path}")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--coins", default=" ".join(CANDIDATES))
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--since", type=str, default="2025-01-01")
    args = ap.parse_args()
    coins = args.coins.split()
    since = date.fromisoformat(args.since)
    merge_tail(args.data, "spot", since, coins, args.workers)
    merge_tail(args.data, "futures/um", since, coins, args.workers)
    merge_funding_tail(args.data, since, coins, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
