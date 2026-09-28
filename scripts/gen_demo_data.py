"""Generate the labeled-SYNTHETIC offline demo dataset into ``data/demo/``.

A fresh clone cannot run the ingest → features → backtest chain without
vendor keys or large downloads.  This script writes a small deterministic
file-based panel that the real ``file`` and ``hf_ohlcv_1m`` providers load:

- ``<out>/daily/raw/{bars,corporate_actions,security_master}.parquet`` —
  canonical bronze-contract inputs for ``dipcatcher ingest`` with
  ``data.source: file`` (see ``configs/demo.yaml``).
- ``<out>/minute_cache/<hf-revision>/ohlcv_YYYY-MM.parquet`` — vendor-shaped
  1-minute OHLCV month cache for ``data.source: hf_ohlcv_1m``
  (see ``configs/demo_minute.yaml``).
- ``<out>/README.md`` and ``<out>/manifest.json`` — SYNTHETIC banner plus
  seed, row counts, and sha256 digests.

The daily panel is produced by ``SyntheticMarketProvider`` — the same
labeled factor model used by ``data.source: synthetic`` — with the
delisting asset dropped (so a ``dipcatcher backtest`` demo cannot strand an
unvalued post-delist hold) and ``ingested_time`` pinned to a deterministic
constant so two runs at the same ``--seed`` produce identical rows.

Everything emitted here is SYNTHETIC data: ``source="synthetic"`` and
``revision_id="SYNTHETIC"`` on every canonical row.  Demo runs prove
pipeline correctness only — they are never market evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from quant_fund.data.adapters.hf_ohlcv_1m import DATASET_REVISION
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider

# Provider ``n_assets`` includes the market row plus one deliberately
# delisted name; dropping it leaves ``N_ASSETS - 1`` fully listed symbols.
N_ASSETS = 9
N_DAYS = 140
DEFAULT_SEED = 13
MINUTE_TICKERS = ("S0001", "S0002")
MINUTE_DAYS = 3
ET = ZoneInfo("America/New_York")
RTH_MINUTES = 390  # 09:30 .. 15:59 ET minute opens


def _pin_ingested(frame: pl.DataFrame, ingested: datetime) -> pl.DataFrame:
    """Replace wall-clock ``ingested_time`` with a deterministic constant."""
    if "ingested_time" not in frame.columns:
        return frame
    return frame.with_columns(pl.lit(ingested).alias("ingested_time"))


def _daily_frames(
    seed: int, n_days: int
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, list[date]]:
    """Synthetic factor panel minus the delisted name, deterministic clock."""
    provider = SyntheticMarketProvider(n_assets=N_ASSETS, n_days=n_days, seed=seed)
    bars = provider.get_bars()
    actions = provider.get_corporate_actions()
    master = provider.get_security_master()
    days = list(provider.days)

    delisted = actions.filter(pl.col("action_type") == "delist").get_column("security_id")
    drop_ids = delisted.to_list() if not delisted.is_empty() else []
    if drop_ids:
        bars = bars.filter(~pl.col("security_id").is_in(drop_ids))
        master = master.filter(~pl.col("security_id").is_in(drop_ids))
    actions = actions.filter(pl.col("action_type") != "delist")

    # One small cash dividend exercises the total-return adjustment path.
    dividend_day = days[int(0.35 * len(days))]
    dividend_close = datetime(
        dividend_day.year, dividend_day.month, dividend_day.day, 16, 0, tzinfo=UTC
    )
    dividend = pl.DataFrame(
        [
            {
                "security_id": "SEC_0002",
                "event_time": dividend_close,
                "available_time": dividend_close,
                "ingested_time": dividend_close,
                "source": "synthetic",
                "revision_id": "SYNTHETIC",
                "action_type": "cash_dividend",
                "factor": None,
                "amount": 0.40,
                "new_ticker": None,
            }
        ]
    ).select(actions.columns)
    # Provider columns are all-Null; relaxed concat supertypes to Float64/String.
    actions = pl.concat([actions, dividend], how="vertical_relaxed")

    # Deterministic ingestion clock: one hour after the last bar close.
    last_close = bars.get_column("event_time").max()
    assert isinstance(last_close, datetime)
    ingested = last_close + timedelta(hours=1)
    bars = _pin_ingested(bars, ingested)
    actions = _pin_ingested(actions, ingested)
    master = _pin_ingested(master, ingested)
    return bars, actions, master, days


def _minute_frame(days: list[date], tickers: tuple[str, ...], seed: int) -> pl.DataFrame:
    """Vendor-schema RTH minute slice (``timestamp`` is the UTC minute open)."""
    rng = np.random.default_rng(seed + 1)
    rows: list[dict[str, object]] = []
    for day in days:
        open_et = datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET)
        for ticker in tickers:
            prev_close = 50.0 + 25.0 * float(rng.random())
            for i in range(RTH_MINUTES):
                stamp_utc = (open_et + timedelta(minutes=i)).astimezone(UTC)
                ret = float(rng.normal(0.0, 0.0005))
                opn = prev_close
                cls = opn * float(np.exp(ret))
                high = max(opn, cls) * (1.0 + abs(float(rng.normal(0.0, 0.0002))))
                low = min(opn, cls) * (1.0 - abs(float(rng.normal(0.0, 0.0002))))
                rows.append(
                    {
                        "timestamp": stamp_utc,
                        "open": opn,
                        "high": high,
                        "low": low,
                        "close": cls,
                        "volume": float(rng.integers(500, 50_000)),
                        "ticker": ticker,
                    }
                )
                prev_close = cls
    return pl.DataFrame(rows).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(out: Path, seed: int, n_days: int = N_DAYS) -> dict[str, object]:
    """Write the demo dataset and return the manifest payload."""
    out = Path(out)
    daily_raw = out / "daily" / "raw"
    minute_root = out / "minute_cache" / DATASET_REVISION
    daily_raw.mkdir(parents=True, exist_ok=True)
    minute_root.mkdir(parents=True, exist_ok=True)

    bars, actions, master, days = _daily_frames(seed, n_days)
    written: dict[str, Path] = {}
    for name, frame in (
        ("bars", bars),
        ("corporate_actions", actions),
        ("security_master", master),
    ):
        path = daily_raw / f"{name}.parquet"
        frame.write_parquet(path)
        written[f"daily/raw/{name}.parquet"] = path

    # The HF cache is one parquet per calendar month — split at boundaries.
    start_idx = min(80, max(0, len(days) - MINUTE_DAYS))
    minute_days = days[start_idx : start_idx + MINUTE_DAYS]
    minutes = _minute_frame(minute_days, MINUTE_TICKERS, seed)
    for year, month in sorted({(d.year, d.month) for d in minute_days}):
        month_minutes = minutes.filter(
            (pl.col("timestamp").dt.year() == year) & (pl.col("timestamp").dt.month() == month)
        )
        month_path = minute_root / f"ohlcv_{year:04d}-{month:02d}.parquet"
        month_minutes.write_parquet(month_path)
        written[f"minute_cache/{DATASET_REVISION}/{month_path.name}"] = month_path

    manifest: dict[str, object] = {
        "kind": "demo_data_manifest",
        "data_label": "SYNTHETIC",
        "claim": (
            "Generated labeled-SYNTHETIC data for offline correctness testing "
            "of ingest/features/backtest. Not market evidence."
        ),
        "generator": "scripts/gen_demo_data.py",
        "seed": int(seed),
        "n_daily_symbols": int(bars.get_column("security_id").n_unique()),
        "n_daily_sessions": len(days),
        "minute_tickers": list(MINUTE_TICKERS),
        "hf_dataset_revision": DATASET_REVISION,
        "files": {
            rel: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for rel, path in sorted(written.items())
        },
        "rows": {
            "daily_bars": bars.height,
            "corporate_actions": actions.height,
            "security_master": master.height,
            "minute_bars": minutes.height,
        },
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    (out / "README.md").write_text(
        "# SYNTHETIC demo data — generated\n"
        "\n"
        "Everything under `data/demo/` is produced by "
        "`uv run python scripts/gen_demo_data.py` (see `make demo-data`) and is "
        'labeled SYNTHETIC (`source="synthetic"`, `revision_id="SYNTHETIC"`).\n'
        "\n"
        "- `daily/raw/` — canonical bronze-contract inputs loaded by the `file` "
        "provider (`configs/demo.yaml`).\n"
        "- `minute_cache/` — vendor-shaped 1-minute OHLCV month cache loaded by "
        "the `hf_ohlcv_1m` provider (`configs/demo_minute.yaml`).\n"
        "- `manifest.json` — seed, row counts, sha256 digests.\n"
        "\n"
        "Results on this data prove pipeline correctness only — never market "
        "evidence. Regenerate; do not commit (the directory is gitignored).\n"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/demo"), help="Output root.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic seed.")
    parser.add_argument(
        "--days",
        type=int,
        default=N_DAYS,
        help="Number of daily sessions (provider minimum is 16).",
    )
    args = parser.parse_args(argv)
    manifest = generate(args.out, seed=args.seed, n_days=args.days)
    rows = manifest["rows"]
    assert isinstance(rows, dict)
    print(f"SYNTHETIC demo data written under {args.out}/")
    for name, count in rows.items():
        print(f"  {name}: {count} rows")
    print(f"  manifest: {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
