"""Fetch Binance public-data-archive carry panels from S3 (data.binance.vision).

Per symbol: USDⓈ-M perp daily klines + spot daily klines + funding-rate events.
Output mirrors ``run_carry_backtest`` inputs:

- data/binance_carry/perp_bars.parquet  (event_time, security_id, ohlcv)
- data/binance_carry/spot_bars.parquet  (same)
- data/binance_carry/funding.parquet    (event_time, security_id, value)

security_id is the base coin (BTC) so BTCUSDT perp pairs with BTCUSDT spot.
Research data collection only — public archive, no auth, no trading claim.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import time
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import polars as pl

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

# Liquid majors as candidate universe (filtered by archive coverage later).
CANDIDATES = [
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "DOGE",
    "BNB",
    "ADA",
    "LINK",
    "LTC",
    "BCH",
    "AVAX",
    "DOT",
    "NEAR",
    "UNI",
    "ATOM",
    "FIL",
    "APT",
    "ARB",
    "OP",
    "INJ",
    "SUI",
    "SEI",
    "TIA",
    "PEPE",
    "WIF",
    "SHIB",
    "TRX",
    "ETC",
    "HBAR",
    "TON",
    "RENDER",
    "FET",
    "GRT",
    "IMX",
    "STX",
    "LDO",
    "CRV",
    "AAVE",
    "MKR",
    "SNX",
    "SAND",
    "MANA",
    "AXS",
    "THETA",
    "FTM",
    "ALGO",
    "VET",
    "ICP",
    "EOS",
    "XLM",
    "KAVA",
    "RUNE",
    "GMX",
    "DYDX",
    "BLUR",
    "JUP",
    "PYTH",
    "ONDO",
    "ENA",
    "W",
    "TAO",
    "NOT",
    "BOME",
    "FLOKI",
    "BONK",
    "ORDI",
]


def _http(url: str, *, retries: int = 6, binary: bool = False):
    delay = 0.4
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "dipcatcher-research/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read() if binary else r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise FileNotFoundError(url) from e
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 8.0)
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 8.0)
    raise RuntimeError("unreachable")


def list_keys(prefix: str, marker: str | None = None) -> list[str]:
    keys: list[str] = []
    marker = marker or ""
    while True:
        url = f"{S3}/?prefix={prefix}&max-keys=1000"
        if marker:
            url += f"&marker={marker}"
        root = ET.fromstring(_http(url))
        batch = [c.find(f"{NS}Key").text for c in root.iter(f"{NS}Contents")]
        keys.extend(batch)
        trunc = root.find(f"{NS}IsTruncated").text == "true"
        if not trunc or not batch:
            break
        marker = batch[-1]
    return keys


def _months_from_keys(keys: list[str]) -> list[str]:
    months = set()
    for k in keys:
        stem = k.removesuffix(".zip")[-7:]
        if len(stem) == 7 and stem[4] == "-" and stem[:4].isdigit():
            months.add(stem)
    return sorted(months)


def _dl_zip_csv(url: str) -> bytes | None:
    try:
        blob = _http(url, binary=True)
    except FileNotFoundError:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            names = [n for n in z.namelist() if n.endswith(".csv")]
            if not names:
                return None
            return z.read(names[0])
    except zipfile.BadZipFile:
        return None


def _epoch_to_dt(ts: int) -> datetime:
    """Binance kline epoch: microseconds (16 digits) from 2025, ms before."""
    return datetime.fromtimestamp(ts / (1e6 if ts > 1e14 else 1e3), tz=UTC)


def _parse_kline_csv(raw: bytes, coin: str, source: str) -> list[dict]:
    rows: list[dict] = []
    for line in raw.decode().splitlines():
        parts = line.split(",")
        if len(parts) < 6 or not parts[0].isdigit():
            continue  # skip header
        try:
            rows.append(
                {
                    "event_time": _epoch_to_dt(int(parts[0])),
                    "security_id": coin,
                    "open": float(parts[1]),
                    "high": float(parts[2]),
                    "low": float(parts[3]),
                    "close": float(parts[4]),
                    "volume": float(parts[5]),
                    "source": source,
                }
            )
        except ValueError:
            continue
    return rows


def _parse_funding_csv(raw: bytes, coin: str) -> list[dict]:
    rows: list[dict] = []
    for line in raw.decode().splitlines():
        parts = line.split(",")
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        try:
            rows.append(
                {
                    "event_time": _epoch_to_dt(int(parts[0])),
                    "security_id": coin,
                    "value": float(parts[2]),
                }
            )
        except ValueError:
            continue
    return rows


def fetch_symbol(coin: str) -> tuple[str, list[dict], list[dict], list[dict]]:
    sym = f"{coin}USDT"
    pk = f"data/futures/um/monthly/klines/{sym}/1d/"
    sk = f"data/spot/monthly/klines/{sym}/1d/"
    fk = f"data/futures/um/monthly/fundingRate/{sym}/"
    p_months = set(_months_from_keys(list_keys(pk)))
    s_months = set(_months_from_keys(list_keys(sk)))
    f_months = set(_months_from_keys(list_keys(fk)))
    if not p_months or not s_months:
        return coin, [], [], []
    all_months = sorted(p_months | s_months | f_months)
    perp_rows: list[dict] = []
    spot_rows: list[dict] = []
    fund_rows: list[dict] = []
    for m in all_months:
        if m in p_months:
            raw = _dl_zip_csv(f"{S3}/{pk}{sym}-1d-{m}.zip")
            if raw:
                perp_rows.extend(_parse_kline_csv(raw, coin, "binance"))
        if m in s_months:
            raw = _dl_zip_csv(f"{S3}/{sk}{sym}-1d-{m}.zip")
            if raw:
                spot_rows.extend(_parse_kline_csv(raw, coin, "binance"))
        if m in f_months:
            raw = _dl_zip_csv(f"{S3}/{fk}{sym}-fundingRate-{m}.zip")
            if raw:
                fund_rows.extend(_parse_funding_csv(raw, coin))
    return coin, perp_rows, spot_rows, fund_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coins", default=" ".join(CANDIDATES), help="space-separated base coins")
    ap.add_argument("--out", type=Path, default=Path("data/binance_carry"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    coins = args.coins.split()
    print(f"fetching {len(coins)} candidate coins")
    perps: list[dict] = []
    spots: list[dict] = []
    funds: list[dict] = []
    keep: list[str] = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_symbol, c): c for c in coins}
        for fut in cf.as_completed(futs):
            coin = futs[fut]
            try:
                _, p, s, f = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  {coin}: FAILED {e}")
                continue
            if not p or not s or not f:
                print(f"  {coin}: incomplete p={len(p)} s={len(s)} f={len(f)}")
                continue
            keep.append(coin)
            perps.extend(p)
            spots.extend(s)
            funds.extend(f)
            print(f"  {coin}: perp={len(p)} spot={len(s)} fund={len(f)}")
    args.out.mkdir(parents=True, exist_ok=True)
    schema_b = {
        "event_time": pl.Datetime("us", UTC),
        "security_id": pl.String,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "source": pl.String,
    }
    schema_f = {
        "event_time": pl.Datetime("us", UTC),
        "security_id": pl.String,
        "value": pl.Float64,
    }
    perp_df = (
        pl.DataFrame(perps, schema=schema_b)
        .unique(["event_time", "security_id"])
        .sort("event_time")
    )
    spot_df = (
        pl.DataFrame(spots, schema=schema_b)
        .unique(["event_time", "security_id"])
        .sort("event_time")
    )
    fund_df = (
        pl.DataFrame(funds, schema=schema_f)
        .unique(["event_time", "security_id"])
        .sort("event_time")
    )
    perp_df.write_parquet(args.out / "perp_bars.parquet")
    spot_df.write_parquet(args.out / "spot_bars.parquet")
    fund_df.write_parquet(args.out / "funding.parquet")
    print(
        f"wrote {args.out}: perp={perp_df.height} spot={spot_df.height} "
        f"funding={fund_df.height} coins={len(keep)}"
    )
    print("kept:", ",".join(sorted(keep)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
