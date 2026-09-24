"""Fetch daily OHLCV+adjclose for a multi-asset universe via Yahoo v8 chart API.

Writes ``<outdir>/<sym>_1d.parquet`` with columns
``event_time, open, high, low, close, volume, adjclose, security_id``.
OHLC columns are ADJUSTED (scaled by adjclose/close) so bar-to-bar returns
carry dividends/splits — standard for ETF-level books. ``adjclose`` column
kept alongside for provenance.

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

UNIVERSE: dict[str, str] = {
    # equity beta / styles
    "SPY": "eq",
    "QQQ": "eq",
    "DIA": "eq",
    "IWM": "eq",
    "EFA": "eq",
    "EEM": "eq",
    "VEA": "eq",
    "MTUM": "eq",
    "USMV": "eq",
    "QUAL": "eq",
    "VLUE": "eq",
    # equity sectors
    "XLF": "eqsec",
    "XLE": "eqsec",
    "XLK": "eqsec",
    "XLV": "eqsec",
    "XLP": "eqsec",
    "XLU": "eqsec",
    "XLI": "eqsec",
    "XLB": "eqsec",
    "XLY": "eqsec",
    "XLC": "eqsec",
    "XLRE": "eqsec",
    # real estate
    "VNQ": "reit",
    "IYR": "reit",
    # rates / credit
    "TLT": "bond",
    "IEF": "bond",
    "IEI": "bond",
    "SHY": "bond",
    "BIL": "bond",
    "LQD": "bond",
    "HYG": "bond",
    "TIP": "bond",
    "EMB": "bond",
    "MUB": "bond",
    "AGG": "bond",
    "VCIT": "bond",
    "BND": "bond",
    "TLH": "bond",
    # commodities
    "GLD": "cmd",
    "SLV": "cmd",
    "USO": "cmd",
    "UNG": "cmd",
    "DBC": "cmd",
    "PPLT": "cmd",
    "CPER": "cmd",
    "DBB": "cmd",
    # fx
    "UUP": "fx",
    "FXE": "fx",
    "FXY": "fx",
    "FXB": "fx",
    "FXF": "fx",
    "FXA": "fx",
    # vol risk premium (short-vol sleeve; DD-governed)
    "SVXY": "vol",
    "VXX": "vol",
    # indices (features/long history)
    "^GSPC": "idx",
    "^IXIC": "idx",
    "^DJI": "idx",
    "^RUT": "idx",
    "^VIX": "idx",
    "^TNX": "idx",
    "^IRX": "idx",
    "^FVX": "idx",
}


def fetch_chart(sym: str, timeout: int = 20) -> dict:
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        + urllib.request.quote(sym, safe="")
        + "?interval=1d&period1=0&period2=4102444800&events=div%2Csplit"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def to_frame(payload: dict, sym: str) -> pl.DataFrame:
    res = payload["chart"]["result"][0]
    ts = res.get("timestamp") or []
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    n = len(ts)
    if n == 0:
        return pl.DataFrame()
    opens = q.get("open") or [None] * n
    highs = q.get("high") or [None] * n
    lows = q.get("low") or [None] * n
    closes = q.get("close") or [None] * n
    vols = q.get("volume") or [None] * n
    adjc = adj or closes
    rows = []
    for t, o, h, lo, c, v, a in zip(ts, opens, highs, lows, closes, vols, adjc, strict=True):
        if c is None or a is None or a <= 0 or c <= 0:
            continue
        f = float(a) / float(c)
        rows.append(
            (
                int(t) * 1_000_000_000,
                (float(o) * f if o else None),
                (float(h) * f if h else None),
                (float(lo) * f if lo else None),
                float(a),
                (float(v) if v else None),
                float(a),
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
    ap.add_argument("--sleep", type=float, default=1.2)
    ap.add_argument("--syms", type=str, default=None, help="comma subset")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    syms = args.syms.split(",") if args.syms else list(UNIVERSE)
    manifest = {}
    for sym in syms:
        dest = args.outdir / f"{sym.replace('^', '').lower()}_1d.parquet"
        try:
            df = to_frame(fetch_chart(sym), sym)
        except Exception:  # noqa: BLE001 - retry once then record failure
            time.sleep(2.0)
            try:
                df = to_frame(fetch_chart(sym), sym)
            except Exception as e2:  # noqa: BLE001
                manifest[sym] = {"error": str(e2), "class": UNIVERSE.get(sym)}
                print(f"{sym}: FAIL {e2}", flush=True)
                continue
        if df.is_empty():
            manifest[sym] = {"error": "empty", "class": UNIVERSE.get(sym)}
            continue
        df.write_parquet(dest)
        manifest[sym] = {
            "rows": df.height,
            "first": str(df["event_time"][0]),
            "last": str(df["event_time"][-1]),
            "class": UNIVERSE.get(sym),
            "file": dest.name,
        }
        print(f"{sym}: {df.height} rows {df['event_time'][0]}..{df['event_time'][-1]}", flush=True)
        time.sleep(args.sleep)
    (args.outdir / "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
