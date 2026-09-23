"""Deep Binance public-klines collection via the repo's source adapter.

``dipcatcher collect`` fetches at most 1000 bars per call (Binance API limit).
This script paginates ``start_time``/``end_time`` backwards through the same
``BinancePublicDataSource`` adapter, concatenates, de-duplicates on
``event_time``, and persists through ``write_source_frame`` so the parquet +
sha256/PIT receipt format matches ordinary collection exactly.

Example:
    .venv/bin/python scripts/collect_binance_deep.py \
        --symbol BTCUSDT --interval 1d --max-bars 4000 --root data
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from quant_fund.data.sources.adapters import BinancePublicDataSource
from quant_fund.data.sources.base import SourceError
from quant_fund.data.sources.storage import write_source_frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--max-bars", type=int, default=4000)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--filename", help="default <symbol>_<interval>_deep.parquet")
    args = parser.parse_args()

    adapter = BinancePublicDataSource()
    frames: list[pl.DataFrame] = []
    end_time: int | None = None
    prev_earliest: int | None = None
    while sum(f.height for f in frames) < args.max_bars:
        try:
            batch = adapter.fetch(
                symbol=args.symbol, interval=args.interval, end_time=end_time, limit=1000
            )
        except SourceError:
            # fetch() raises on an empty page — that is the listing boundary.
            if frames:
                break
            raise
        if batch.is_empty():
            break
        frames.append(batch)
        # The adapter drops in-progress bars, so a full page can read as 999
        # rows; terminate on no progress (listing boundary) rather than count.
        earliest = int(batch["event_time"].cast(pl.Int64).min())
        if prev_earliest is not None and earliest >= prev_earliest:
            break
        prev_earliest = earliest
        end_time = earliest - 1
        time.sleep(0.3)  # polite pacing on a public endpoint

    if not frames:
        raise SystemExit(f"{args.symbol} {args.interval}: no rows fetched")
    frame = (
        pl.concat(frames)
        .unique(subset=["event_time"], keep="first")
        .sort("event_time")
        .tail(args.max_bars)
    )
    filename = args.filename or f"{args.symbol.lower()}_{args.interval}_deep.parquet"
    paths = write_source_frame(
        frame,
        args.root,
        BinancePublicDataSource.name,
        filename=filename,
        provenance={
            "request": {
                "symbol": args.symbol.upper(),
                "interval": args.interval,
                "max_bars": args.max_bars,
                "paginated": True,
            }
        },
    )
    print(f"rows={frame.height} data={paths['data']} receipt={paths['receipt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
