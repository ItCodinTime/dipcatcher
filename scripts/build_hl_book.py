"""Build the Hyperliquid carry book: resolve each perp's spot hedge leg.

Reads raw fetches:
- data/hyperliquid_carry/perp_bars.parquet    (HL perp daily candles)
- data/hyperliquid_carry/hl_spot_bars.parquet (HL-native spot candles)
- data/hyperliquid_carry/funding.parquet      (hourly funding events)
- data/binance_carry/spot_bars.parquet + data/binance_carry_extra/spot_bars.parquet
  (Binance spot, used as the hedge leg for HL perps without an HL spot pair)

Emits data/hl_carry_book/{perp_bars,spot_bars,funding}.parquet where every
security_id is prefixed ``HL:`` — overlapping coins (BTC on both venues) become
distinct book names (BTC vs HL:BTC), which is honest: they are separate
delta-neutral positions on different venues, each marked at its own venue's
wicks for liquidation.

Merge into the Binance universe with ``carry_research.py --extra-dir
data/hl_carry_book``; run standalone with ``--data-dir data/hl_carry_book``.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

HL = Path("data/hyperliquid_carry")
BINANCE_DIRS = [Path("data/binance_carry"), Path("data/binance_carry_extra")]


def _load(name: str) -> pl.DataFrame:
    frames = [pl.read_parquet(d / name) for d in BINANCE_DIRS if (d / name).exists()]
    return pl.concat(frames).unique(["event_time", "security_id"]) if frames else pl.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/hl_carry_book"))
    args = ap.parse_args()

    perp = pl.read_parquet(HL / "perp_bars.parquet")
    hl_spot = pl.read_parquet(HL / "hl_spot_bars.parquet")
    fund = pl.read_parquet(HL / "funding.parquet")
    today = datetime.now(tz=UTC).date()
    perp = perp.filter(pl.col("event_time").dt.date() < today)
    hl_spot = hl_spot.filter(pl.col("event_time").dt.date() < today)
    bin_spot = _load("spot_bars.parquet")

    bin_ids = set(bin_spot["security_id"].unique()) if bin_spot.height else set()
    hl_spot_ids = set(hl_spot["security_id"].unique()) if hl_spot.height else set()
    coins = sorted(set(perp["security_id"].unique()))

    out_perp, out_spot, out_fund = [], [], []
    stats = {"hl_spot": 0, "binance_spot": 0, "no_leg": 0}
    for coin in coins:
        p = perp.filter(pl.col("security_id") == coin)
        if not p.height:
            continue
        if coin in hl_spot_ids:
            s = hl_spot.filter(pl.col("security_id") == coin)
            stats["hl_spot"] += 1
        elif coin in bin_ids:
            s = bin_spot.filter(pl.col("security_id") == coin)
            stats["binance_spot"] += 1
        else:
            stats["no_leg"] += 1
            continue
        sid = f"HL:{coin}"
        out_perp.append(p.with_columns(pl.lit(sid).alias("security_id")))
        out_spot.append(s.with_columns(pl.lit(sid).alias("security_id")))
        f = fund.filter(pl.col("security_id") == coin)
        if f.height:
            out_fund.append(f.with_columns(pl.lit(sid).alias("security_id")))

    args.out.mkdir(parents=True, exist_ok=True)
    for name, frames in (
        ("perp_bars", out_perp),
        ("spot_bars", out_spot),
        ("funding", out_fund),
    ):
        df = (
            pl.concat(frames).unique(["event_time", "security_id"]).sort("event_time")
            if frames
            else pl.DataFrame()
        )
        df.write_parquet(args.out / f"{name}.parquet")
        print(f"{name}: {df.height} rows")
    print(f"coins: {sum(1 for c in coins if c in hl_spot_ids or c in bin_ids)} | {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
