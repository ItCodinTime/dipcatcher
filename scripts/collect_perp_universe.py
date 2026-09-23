"""Megaplan Phase A: collect a top-N Binance USDT-M perp universe.

Resolves the current USDT-M PERPETUAL universe ranked by 24h quote volume,
then collects (per symbol, resumable):

- ``{symbol}_{interval}.perp.parquet`` for each ``--interval`` (default 1h 4h 1d)
- ``{symbol}.funding.parquet`` funding-rate history (``--funding``, default on)
- ``{symbol}_{interval}.spot.parquet`` spot klines over the same window
  (``--spot``, for the conservative spot-only book)

Writes ``perp_universe.parquet`` + ``collection_manifest.json`` with per-file
sha256, row counts, first/last event_time, and a gap audit (observed vs
expected bars between first and last open). Per-symbol start is the
``onboardDate`` from exchangeInfo so listing truncation is explicit.

Survivorship: the public API only lists currently-traded symbols. That bias is
disclosed in the manifest — it cannot be removed with public data.

Research collection only; uses the network explicitly (never via ingest).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import polars as pl  # noqa: E402

from quant_fund.data.collector import collect_source  # noqa: E402
from quant_fund.data.sources.base import SourceError  # noqa: E402

_INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
    "1w": 604_800_000,
}
# Fiat/stable base assets dilute the cross-section with ~zero-vol names.
_DEFAULT_EXCLUDE_BASES = {
    "USDC",
    "FDUSD",
    "TUSD",
    "DAI",
    "USDP",
    "BFUSD",
    "USD1",
    "USDE",
    "XUSD",
    "EUR",
    "GBP",
    "AUD",
    "BRL",
    "TRY",
    "AEUR",
    "EURI",
    "GYEN",
    "IDRT",
    "BIDR",
    "ARSD",
    "UAH",
    "NGN",
    "ZAR",
    "MXN",
    "COP",
    "JPY",
    "RUB",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit(path: Path, interval: str | None) -> dict:
    frame = pl.read_parquet(path)
    if frame.is_empty():
        return {"rows": 0}
    times = frame["event_time"].sort()
    entry = {
        "rows": frame.height,
        "first_event": str(times[0]),
        "last_event": str(times[-1]),
        "sha256": _sha256(path),
    }
    if interval and interval in _INTERVAL_MS and frame.height >= 2:
        step = _INTERVAL_MS[interval]
        epoch_ms = [int(t.timestamp() * 1000) for t in times.to_list()]
        expected = (epoch_ms[-1] - epoch_ms[0]) // step + 1
        gaps = sum(max(0, (b - a) // step - 1) for a, b in itertools.pairwise(epoch_ms))
        entry["expected_bars_in_span"] = int(expected)
        entry["missing_bars_in_span"] = int(gaps)
        entry["missing_pct"] = round(100.0 * gaps / max(expected, 1), 4)
    return entry


def _collect_with_retries(
    source: str, root: Path, fetch_kwargs: dict, filename: str, retries: int = 3
) -> Path:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            result = collect_source(source, root, fetch_kwargs=fetch_kwargs, filename=filename)
            return result.data
        except (SourceError, OSError, ValueError) as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(30.0 * (attempt + 1))
    raise SourceError(f"{filename}: failed after {retries} attempts: {last}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--intervals", nargs="+", default=["1h", "4h", "1d"])
    parser.add_argument("--data-root", type=Path, default=_REPO / "data")
    parser.add_argument("--min-quote-vol", type=float, default=5_000_000.0)
    parser.add_argument("--exclude-bases", nargs="*", default=sorted(_DEFAULT_EXCLUDE_BASES))
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Explicit symbol allowlist (skips universe resolution)",
    )
    parser.add_argument(
        "--symbols-file",
        type=Path,
        default=None,
        help="Text file with one symbol per line (alternative to --symbols)",
    )
    parser.add_argument("--funding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--spot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--spot-intervals", nargs="+", default=["1h", "1d"])
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument("--pause", type=float, default=0.3)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    data_root = args.data_root
    sources_dir = data_root / "raw" / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (sources_dir / "collection_manifest.json")

    from quant_fund.data.sources.adapters import BinancePerpUniverseSource

    symbols_arg = args.symbols
    if symbols_arg is None and args.symbols_file is not None:
        symbols_arg = [
            line.strip()
            for line in args.symbols_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if symbols_arg:
        # Explicit allowlist still needs onboard dates: resolve the full table.
        uni = BinancePerpUniverseSource().fetch()
        allowed = {s.upper() for s in symbols_arg}
        uni = uni.filter(pl.col("security_id").is_in(allowed))
        if uni.is_empty():
            raise SystemExit("no requested symbols are listed USDT-M PERPETUALs")
    else:
        uni = BinancePerpUniverseSource().fetch(min_quote_volume=float(args.min_quote_vol))
        uni = uni.filter(~pl.col("base_asset").is_in(list(args.exclude_bases)))
        uni = uni.head(args.top_n)
        # Re-rank after filtering so the manifest rank is the effective one.
        uni = uni.with_columns((pl.int_range(1, pl.len() + 1)).alias("rank"))

    uni_path = sources_dir / "perp_universe.parquet"
    if symbols_arg and uni_path.exists():
        # Shard runs must not overwrite the canonical full-universe file.
        print(f"universe allowlist: {uni.height} of {len(allowed)} requested")
    else:
        uni.write_parquet(uni_path)
        print(f"universe: {uni.height} symbols -> {uni_path}")
    symbols = uni["security_id"].to_list()
    onboard = {
        str(r["security_id"]): int(r["onboard_ms"])
        for r in uni.select(["security_id", "onboard_ms"]).iter_rows(named=True)
    }


    jobs: list[tuple[str, str, dict, str]] = []
    for symbol in symbols:
        start_ms = onboard[symbol]
        for interval in args.intervals:
            jobs.append(
                (
                    "binance_perp",
                    f"{symbol.lower()}_{interval}.perp.parquet",
                    {
                        "symbol": symbol,
                        "interval": interval,
                        "start_time": start_ms,
                        "limit": args.limit,
                        "max_pages": args.max_pages,
                        "pause_seconds": args.pause,
                    },
                    interval,
                )
            )
        if args.funding:
            jobs.append(
                (
                    "binance_funding",
                    f"{symbol.lower()}.funding.parquet",
                    {
                        "symbol": symbol,
                        "start_time": start_ms,
                        "limit": 1000,
                        "pause_seconds": args.pause,
                    },
                    None,
                )
            )
        if args.spot:
            for interval in args.spot_intervals:
                jobs.append(
                    (
                        "binance",
                        f"{symbol.lower()}_{interval}.spot.parquet",
                        {
                            "symbol": symbol,
                            "interval": interval,
                            "start_time": start_ms,
                            "limit": 1000,
                            "max_pages": args.max_pages,
                            "pause_seconds": args.pause,
                        },
                        interval,
                    )
                )

    if args.dry_run:
        print(f"dry-run: {len(jobs)} jobs over {len(symbols)} symbols")
        return 0

    manifest: dict = {
        "schema": "perp_collection.v1",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "top_n": args.top_n,
        "intervals": args.intervals,
        "spot_intervals": args.spot_intervals if args.spot else [],
        "funding": args.funding,
        "min_quote_volume": args.min_quote_vol,
        "exclude_bases": args.exclude_bases,
        "survivorship": (
            "Universe = symbols listed as TRADING USDT-M PERPETUAL at collection "
            "time; delisted symbols are absent (public API limitation, disclosed)."
        ),
        "universe_file": str(uni_path),
        "universe_sha256": _sha256(uni_path),
        "n_symbols": len(symbols),
        "symbols": symbols,
        "files": {},
        "failures": {},
        "skipped_existing": [],
    }

    t0 = time.time()
    for idx, (source, filename, kwargs, interval) in enumerate(jobs, start=1):
        target = sources_dir / filename
        if args.resume and target.exists() and target.stat().st_size > 0:
            manifest["skipped_existing"].append(filename)
            continue
        try:
            path = _collect_with_retries(source, data_root, kwargs, filename)
            manifest["files"][filename] = _audit(path, interval)
            if idx % 20 == 0 or idx == len(jobs):
                done = len(manifest["files"])
                rate = done / max(time.time() - t0, 1e-9)
                print(
                    f"[{idx}/{len(jobs)}] collected={done} "
                    f"failures={len(manifest['failures'])} rate={rate:.2f} files/s"
                )
        except SourceError as exc:
            manifest["failures"][filename] = str(exc)
            print(f"FAILED {filename}: {exc}", file=sys.stderr)
        # Persist after every file so a crash loses at most one entry.
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    manifest["elapsed_seconds"] = round(time.time() - t0, 1)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"done: {len(manifest['files'])} files, "
        f"{len(manifest['failures'])} failures, "
        f"{len(manifest['skipped_existing'])} skipped -> {manifest_path}"
    )
    return 1 if manifest["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
