"""Build the cross-venue funding-spread arb book at 1h price grain.

Same ARBX/ARBB construction as ``build_arb_book.py`` but on hourly bars:
HL perp candles from ``data/hl_carry_1h`` and Binance perp klines from
``data/binance_carry_1h`` (+ ``binance_carry`` funding).

Funding spread is emitted once per Binance funding epoch (00/08/16 UTC):
``value = Σ HL hourly funding over the 8h epoch window − r_Binance`` for
ARBX (short HL / long Binance), the mirror sign for ARBB. Epochs missing
either venue are skipped — an unverifiable spread is not zero.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

HL = Path("data/hl_carry_1h")
HL_FUND = Path("data/hyperliquid_carry")
BINANCE = Path("data/binance_carry_1h")
BIN_FUND_DIRS = [Path("data/binance_carry_1h"), Path("data/binance_carry")]
OUT = Path("data/arb_carry_book_1h")

_BARS = ("event_time", "security_id", "open", "high", "low", "close", "volume", "source")
_BIN_EPOCH_H = {0, 8, 16}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--hl", default=str(HL))
    ap.add_argument("--binance", default=str(BINANCE))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    hl_perp = pl.read_parquet(Path(args.hl) / "perp_bars.parquet")
    bin_perp = pl.read_parquet(Path(args.binance) / "perp_bars.parquet")
    coins = sorted(set(hl_perp["security_id"].unique()) & set(bin_perp["security_id"].unique()))
    print(f"common coins: {len(coins)}")

    # Binance funding: one event per (coin, epoch) — epochs already at 00/08/16.
    bin_f = (
        pl.concat(
            [
                pl.read_parquet(d / "funding.parquet")
                for d in BIN_FUND_DIRS
                if (d / "funding.parquet").exists()
            ]
        )
        .unique(["event_time", "security_id"], keep="first")
        .filter(pl.col("event_time").dt.hour().is_in(list(_BIN_EPOCH_H)))
        .rename({"value": "r_bin"})
    )
    # HL funding: hourly, paid at the top of each hour for the hour just
    # ended, but event timestamps carry API jitter (seconds past the hour).
    # Round each event to its payment hour boundary, then bucket boundaries
    # into the Binance 8h epoch they belong to: boundary B ∈ (E−8h, E].
    hl_f = (
        pl.read_parquet(HL_FUND / "funding.parquet")
        .with_columns(
            (
                (pl.col("event_time").dt.round("1h") - pl.duration(milliseconds=1)).dt.truncate(
                    "8h"
                )
                + pl.duration(hours=8)
            ).alias("epoch")
        )
        .group_by("security_id", "epoch")
        .agg(pl.col("value").sum().alias("r_hl"))
        .rename({"epoch": "event_time"})
    )
    spread = hl_f.join(bin_f, on=["security_id", "event_time"], how="inner")

    perp_rows: list[pl.DataFrame] = []
    spot_rows: list[pl.DataFrame] = []
    fund_rows: list[pl.DataFrame] = []
    kept = 0
    for c in coins:
        hp = hl_perp.filter(pl.col("security_id") == c).select(*_BARS)
        bp = bin_perp.filter(pl.col("security_id") == c).select(*_BARS)
        sp = spread.filter(pl.col("security_id") == c)
        # Both directions need both venues' marks plus verifiable funding:
        # clip to the intersection of all three panels' coverage.
        lo_ts = max(hp["event_time"].min(), bp["event_time"].min(), sp["event_time"].min())
        hi_ts = min(hp["event_time"].max(), bp["event_time"].max(), sp["event_time"].max())
        hp = hp.filter(pl.col("event_time").is_between(lo_ts, hi_ts))
        bp = bp.filter(pl.col("event_time").is_between(lo_ts, hi_ts))
        if hp.height < 480 or bp.height < 480 or sp.height < 180:
            continue
        kept += 1
        perp_rows.append(hp.with_columns(security_id=pl.lit(f"ARBX:{c}")))
        spot_rows.append(bp.with_columns(security_id=pl.lit(f"ARBX:{c}")))
        fund_rows.append(
            sp.select(
                "event_time",
                pl.lit(f"ARBX:{c}").alias("security_id"),
                (pl.col("r_hl") - pl.col("r_bin")).alias("value"),
            )
        )
        perp_rows.append(bp.with_columns(security_id=pl.lit(f"ARBB:{c}")))
        spot_rows.append(hp.with_columns(security_id=pl.lit(f"ARBB:{c}")))
        fund_rows.append(
            sp.select(
                "event_time",
                pl.lit(f"ARBB:{c}").alias("security_id"),
                (pl.col("r_bin") - pl.col("r_hl")).alias("value"),
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
