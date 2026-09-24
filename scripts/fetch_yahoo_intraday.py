"""Fetch intraday (hourly) OHLCV via Yahoo v8 chart API.

Yahoo limits ``interval=1h`` to ~730 days of history — the honest tradeoff
for ~7x more independent decisions per year vs daily bars.

Writes ``<outdir>/<sym>_1h.parquet`` with the same schema as the daily
fetcher (event_time/open/high/low/close/volume/adjclose/security_id).
Bars are split-adjusted by Yahoo; dividend gaps inside the 2y window are
a disclosed limitation of public intraday data.

No API key; polite serial fetch with retry. Research-only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import polars as pl

# Liquid names only — intraday signals need tight spreads. ETFs + megacap
# stocks that trade through the full session.
SYMS: list[str] = [
    # broad + sector ETFs
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "XLF",
    "XLE",
    "XLK",
    "XLV",
    "XLP",
    "XLU",
    "XLI",
    "XLB",
    "XLY",
    "XLRE",
    "TLT",
    "IEF",
    "SHY",
    "LQD",
    "HYG",
    "TIP",
    "GLD",
    "SLV",
    "USO",
    "UNG",
    "UUP",
    "FXE",
    "FXY",
    "EEM",
    "EFA",
    "VNQ",
    "SVXY",
    "VXX",
    # liquid megacap stocks
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AVGO",
    "JPM",
    "V",
    "MA",
    "XOM",
    "CVX",
    "UNH",
    "HD",
    "PG",
    "KO",
    "PEP",
    "COST",
    "WMT",
    "MCD",
    "CRM",
    "AMD",
    "INTC",
    "NFLX",
    "DIS",
    "BA",
    "CAT",
    "GE",
    "HON",
    "UPS",
    "GS",
    "MS",
    "BAC",
    "C",
    "WFC",
    "T",
    "VZ",
    "PFE",
    "MRK",
    "ABT",
    "LLY",
    "JNJ",
    "ORCL",
    "IBM",
    "QCOM",
    "TXN",
    "ADBE",
    "NOW",
    "PANW",
    "CSCO",
    "MU",
    "F",
    "GM",
    "DE",
]


def fetch_chart(sym: str, rng: str, timeout: int = 20) -> dict:
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        + urllib.request.quote(sym, safe="")
        + f"?interval=1h&range={rng}&includePrePost=false"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def to_frame(payload: dict, sym: str) -> pl.DataFrame:
    res = payload["chart"]["result"][0]
    ts = res.get("timestamp") or []
    q = res["indicators"]["quote"][0]
    n = len(ts)
    if n == 0:
        return pl.DataFrame()
    opens = q.get("open") or [None] * n
    highs = q.get("high") or [None] * n
    lows = q.get("low") or [None] * n
    closes = q.get("close") or [None] * n
    vols = q.get("volume") or [None] * n
    rows = []
    for t, o, h, lo, c, v in zip(ts, opens, highs, lows, closes, vols, strict=True):
        if c is None or c <= 0 or o is None or o <= 0:
            continue
        rows.append(
            (
                int(t) * 1_000_000_000,
                float(o),
                float(h) if h else float(c),
                float(lo) if lo else float(c),
                float(c),
                float(v) if v else None,
                float(c),
            )
        )
    if not rows:
        return pl.DataFrame()
    cols = list(zip(*rows, strict=True))
    df = pl.DataFrame(
        {
            "event_time": pl.Series(cols[0], dtype=pl.Int64),
            "open": pl.Series(cols[1], dtype=pl.Float64),
            "high": pl.Series(cols[2], dtype=pl.Float64),
            "low": pl.Series(cols[3], dtype=pl.Float64),
            "close": pl.Series(cols[4], dtype=pl.Float64),
            "volume": pl.Series(cols[5], dtype=pl.Float64),
            "adjclose": pl.Series(cols[6], dtype=pl.Float64),
        }
    )
    return df.with_columns(
        pl.from_epoch("event_time", time_unit="ns").dt.replace_time_zone("UTC"),
        pl.lit(sym).alias("security_id"),
    ).sort("event_time")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--range", type=str, default="730d")
    ap.add_argument("--syms", type=str, default=None)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    syms = args.syms.split(",") if args.syms else SYMS
    manifest = {}
    for sym in syms:
        dest = args.outdir / f"{sym.lower()}_1h.parquet"
        try:
            df = to_frame(fetch_chart(sym, args.range), sym)
        except Exception:  # noqa: BLE001
            time.sleep(2.0)
            try:
                df = to_frame(fetch_chart(sym, args.range), sym)
            except Exception as e2:  # noqa: BLE001
                manifest[sym] = {"error": str(e2)}
                print(f"{sym}: FAIL {e2}", flush=True)
                continue
        if df.is_empty():
            manifest[sym] = {"error": "empty"}
            continue
        df.write_parquet(dest)
        manifest[sym] = {
            "rows": df.height,
            "first": str(df["event_time"][0]),
            "last": str(df["event_time"][-1]),
            "file": dest.name,
        }
        print(f"{sym}: {df.height} rows {df['event_time'][0]}..{df['event_time'][-1]}", flush=True)
        time.sleep(args.sleep)
    (args.outdir / "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
