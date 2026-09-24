"""Build a multi-venue funding-spread arb book from N perp-venue carry books.

Generalizes ``build_arb_book.py`` (HL vs Binance) over any set of venue dirs
shaped ``{perp_bars, spot_bars, funding}.parquet`` with
``(event_time, security_id, value)`` funding rows. For every unordered venue
pair (A, B) and each coin present in BOTH venues' perp bars (>=60 bars each)
with >=60 joint funding days, emit two synthetic perp-vs-perp pairs:

- ``AX:<a>>>{b}:{coin}`` — short A perp / long B perp, collecting
  ``r_A - r_B`` per day. "perp" leg priced off A bars (short-leg liquidation
  marks), "spot" hedge leg off B *perp* bars (same caveat as the 2-venue
  book: the long perp leg can itself be liquidated on extreme moves; the
  unmodeled residual is venue-specific margin, not NAV).
- ``AX:<b>>>{a}:{coin}`` — the mirror collecting ``r_B - r_A``.

Funding rows are the net daily spread (one event/day at 23:59:59) emitted only
on days where BOTH venues have funding events.

Usage:
    uv run python scripts/build_arb_book_multi.py \
        --venue h=data/hyperliquid_carry \
        --venue b=data/binance_carry \
        --venue o=data/okx_carry \
        --out data/arb3_book
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import polars as pl

_BARS = ("event_time", "security_id", "open", "high", "low", "close", "volume", "source")


def _daily_funding(fund: pl.DataFrame) -> pl.DataFrame:
    return (
        fund.with_columns(pl.col("event_time").dt.date().alias("d"))
        .group_by("security_id", "d")
        .agg(pl.col("value").sum().alias("v"))
        .with_columns(pl.col("d").cast(pl.Datetime).dt.replace_time_zone("UTC"))
    )


def _load_venue(d: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    perp = pl.read_parquet(d / "perp_bars.parquet").select(*_BARS)
    fund = _daily_funding(
        pl.read_parquet(d / "funding.parquet").select("event_time", "security_id", "value")
    )
    return perp, fund


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--venue",
        action="append",
        required=True,
        help="code=dir e.g. h=data/hyperliquid_carry (>=2)",
    )
    ap.add_argument("--out", default="data/arb3_book")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from datetime import UTC, datetime

    today = datetime.now(tz=UTC).date()

    venues: dict[str, tuple[pl.DataFrame, pl.DataFrame]] = {}
    for spec in args.venue:
        code, _, dstr = spec.partition("=")
        perp, fund = _load_venue(Path(dstr))
        perp = perp.filter(pl.col("event_time").dt.date() < today)
        venues[code] = (perp, fund)
        print(f"{code} ({dstr}): perp={perp.height} fund_days={fund.height}")

    # Shared calendar: venues fetched on different dates otherwise poison the
    # "still listed at window end" eligibility check for every pair-sid.
    min_end = min(p["event_time"].max() for p, _ in venues.values())
    venues = {
        c: (
            p.filter(pl.col("event_time") <= min_end),
            f.filter(pl.col("d") <= min_end),
        )
        for c, (p, f) in venues.items()
    }
    print("common window end:", min_end)

    perp_rows: list[pl.DataFrame] = []
    spot_rows: list[pl.DataFrame] = []
    fund_rows: list[pl.DataFrame] = []
    kept = 0
    for a, b in itertools.combinations(venues, 2):
        pa, fa = venues[a]
        pb, fb = venues[b]
        coins = sorted(set(pa["security_id"].unique()) & set(pb["security_id"].unique()))
        spread = (
            fa.join(fb, on=["security_id", "d"], how="inner", suffix="_b")
            .with_columns(
                (pl.col("d") + pl.duration(hours=23, minutes=59, seconds=59)).alias("event_time")
            )
            .select("event_time", "security_id", "v", "v_b")
        )
        for c in coins:
            hp = pa.filter(pl.col("security_id") == c)
            bp = pb.filter(pl.col("security_id") == c)
            sp = spread.filter(pl.col("security_id") == c)
            if hp.height < 60 or bp.height < 60 or sp.height < 60:
                continue
            kept += 1
            sid_ab = f"AX:{a}>{b}:{c}"
            sid_ba = f"AX:{b}>{a}:{c}"
            perp_rows.append(hp.with_columns(security_id=pl.lit(sid_ab)))
            spot_rows.append(bp.with_columns(security_id=pl.lit(sid_ab)))
            fund_rows.append(
                sp.select(
                    "event_time",
                    pl.lit(sid_ab).alias("security_id"),
                    (pl.col("v") - pl.col("v_b")).alias("value"),
                )
            )
            perp_rows.append(bp.with_columns(security_id=pl.lit(sid_ba)))
            spot_rows.append(hp.with_columns(security_id=pl.lit(sid_ba)))
            fund_rows.append(
                sp.select(
                    "event_time",
                    pl.lit(sid_ba).alias("security_id"),
                    (pl.col("v_b") - pl.col("v")).alias("value"),
                )
            )

    for name, rows in (("perp_bars", perp_rows), ("spot_bars", spot_rows), ("funding", fund_rows)):
        df = pl.concat(rows).sort(["security_id", "event_time"])
        df.write_parquet(out / f"{name}.parquet")
        print(f"{name}: {df.height} rows")
    print("pair-sids kept:", kept)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
