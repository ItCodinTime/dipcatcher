"""Build the dYdX carry book: attach the Binance spot hedge leg.

Reads raw fetches:
- data/dydx_carry/perp_bars.parquet (dYdX perp daily candles)
- data/dydx_carry/funding.parquet   (hourly funding events)
- Binance spot_bars from data/binance_carry/ + data/binance_carry_extra/
  (dYdX lists no spot markets, so the hedge leg always lives on Binance)

Emits data/dydx_carry_book/{perp_bars,spot_bars,funding}.parquet where every
security_id is prefixed ``DYDX:`` — overlapping coins become distinct book
names (BTC vs DYDX:BTC), i.e. separate delta-neutral positions on different
venues each marked at its own wicks for liquidation.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

DYDX = Path("data/dydx_carry")
BINANCE_DIRS = [Path("data/binance_carry"), Path("data/binance_carry_extra")]


def _load(name: str) -> pl.DataFrame:
    frames = [pl.read_parquet(d / name) for d in BINANCE_DIRS if (d / name).exists()]
    return pl.concat(frames).unique(["event_time", "security_id"]) if frames else pl.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/dydx_carry_book"))
    args = ap.parse_args()

    perp = pl.read_parquet(DYDX / "perp_bars.parquet")
    fund = pl.read_parquet(DYDX / "funding.parquet")
    # drop the live partial day: Binance bars end at yesterday's close, and a
    # 1-day-newer union calendar would fail the still-listed eligibility check
    today = datetime.now(tz=UTC).date()
    perp = perp.filter(pl.col("event_time").dt.date() < today)
    bin_spot = _load("spot_bars.parquet")
    bin_ids = set(bin_spot["security_id"].unique())

    out_perp, out_spot, out_fund = [], [], []
    kept = dropped = 0
    for coin in sorted(set(perp["security_id"].unique())):
        if coin not in bin_ids:
            dropped += 1
            continue
        sid = f"DYDX:{coin}"
        out_perp.append(
            perp.filter(pl.col("security_id") == coin).with_columns(
                pl.lit(sid).alias("security_id")
            )
        )
        out_spot.append(
            bin_spot.filter(pl.col("security_id") == coin).with_columns(
                pl.lit(sid).alias("security_id")
            )
        )
        f = fund.filter(pl.col("security_id") == coin)
        if f.height:
            out_fund.append(f.with_columns(pl.lit(sid).alias("security_id")))
        kept += 1

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
    print(f"coins: {kept} kept | {dropped} dropped (no Binance spot)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
