"""Build a cross-venue funding-spread arb book (Hyperliquid vs Binance perps).

For coins listed on BOTH HL and Binance perps, emit two synthetic pairs:

- ``ARBX:{coin}`` — short HL perp / long Binance perp, collecting
  ``r_HL − r_Binance`` per day. The engine's "perp" leg is priced off HL bars
  (short leg marked at HL wicks for liquidation) and the "spot" hedge leg off
  Binance *perp* bars — the hedge is itself a perp; see the docs for the
  honest caveat (a long perp can itself be liquidated on >~33% single-day
  moves; same-day crashes on both venues leave the net book near flat, so the
  unmodeled residual is venue-specific margin, not NAV).
- ``ARBB:{coin}`` — the mirror: short Binance perp / long HL perp collecting
  ``r_Binance − r_HL``, perp leg priced off Binance bars, hedge off HL bars.

Funding rows are the net daily spread (one event/day at 23:59:59) emitted only
on days where BOTH venues have funding events — missing either side means the
spread is unverifiable that day, not zero.

Output ``data/arb_carry_book/{perp_bars,spot_bars,funding}.parquet``; merge
with ``--extra-dir``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

HL = Path("data/hyperliquid_carry")
BINANCE_DIRS = [Path("data/binance_carry"), Path("data/binance_carry_extra")]
OUT = Path("data/arb_carry_book")

_BARS = ("event_time", "security_id", "open", "high", "low", "close", "volume", "source")


def _load(fname: str) -> pl.DataFrame:
    parts = [pl.read_parquet(d / fname) for d in BINANCE_DIRS if (d / fname).exists()]
    return pl.concat(parts).unique(["event_time", "security_id"], keep="first")


def _daily_funding(fund: pl.DataFrame, prefix: str) -> pl.DataFrame:
    return (
        fund.with_columns(pl.col("event_time").dt.date().alias("d"))
        .group_by("security_id", "d")
        .agg(pl.col("value").sum().alias("v"))
        .select(
            pl.col("d").cast(pl.Datetime).dt.replace_time_zone("UTC").alias("d"),
            pl.col("security_id"),
            pl.col("v"),
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from datetime import UTC, datetime

    today = datetime.now(tz=UTC).date()

    hl_perp = pl.read_parquet(HL / "perp_bars.parquet").filter(
        pl.col("event_time").dt.date() < today
    )
    bin_perp = _load("perp_bars.parquet")
    coins = sorted(set(hl_perp["security_id"].unique()) & set(bin_perp["security_id"].unique()))

    hl_f = _daily_funding(pl.read_parquet(HL / "funding.parquet"), "hl")
    bin_f = _daily_funding(
        pl.concat(
            [
                pl.read_parquet(d / "funding.parquet")
                for d in BINANCE_DIRS
                if (d / "funding.parquet").exists()
            ]
        ).unique(["event_time", "security_id"], keep="first"),
        "bin",
    )
    spread = (
        hl_f.join(bin_f, on=["security_id", "d"], how="inner", suffix="_bin")
        .with_columns(
            (pl.col("d") + pl.duration(hours=23, minutes=59, seconds=59)).alias("event_time")
        )
        .select("event_time", "security_id", "v", "v_bin")
    )

    perp_rows = []
    spot_rows = []
    fund_rows = []
    kept = 0
    for c in coins:
        hp = hl_perp.filter(pl.col("security_id") == c).select(*_BARS)
        bp = bin_perp.filter(pl.col("security_id") == c).select(*_BARS)
        sp = spread.filter(pl.col("security_id") == c)
        if hp.height < 60 or bp.height < 60 or sp.height < 60:
            continue
        kept += 1
        perp_rows.append(hp.with_columns(security_id=pl.lit(f"ARBX:{c}")))
        spot_rows.append(bp.with_columns(security_id=pl.lit(f"ARBX:{c}")))
        fund_rows.append(
            sp.select(
                "event_time",
                pl.lit(f"ARBX:{c}").alias("security_id"),
                (pl.col("v") - pl.col("v_bin")).alias("value"),
            )
        )
        perp_rows.append(bp.with_columns(security_id=pl.lit(f"ARBB:{c}")))
        spot_rows.append(hp.with_columns(security_id=pl.lit(f"ARBB:{c}")))
        fund_rows.append(
            sp.select(
                "event_time",
                pl.lit(f"ARBB:{c}").alias("security_id"),
                (pl.col("v_bin") - pl.col("v")).alias("value"),
            )
        )

    for name, rows in (("perp_bars", perp_rows), ("spot_bars", spot_rows), ("funding", fund_rows)):
        df = pl.concat(rows).sort(["security_id", "event_time"])
        df.write_parquet(out / f"{name}.parquet")
        print(f"{name}: {df.height} rows")
    print("coins:", kept, "| sids:", kept * 2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
