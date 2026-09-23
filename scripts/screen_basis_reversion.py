"""Development-only screen for cross-sectional spot/perp basis reversion.

At completed close t, an expensive perp versus its own trailing 30-bar basis
is shorted and a cheap perp is bought. The outcome is next bar open-to-close
perp return. This is a pre-cost diagnostic, not a trading backtest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sources = args.data_root / "raw" / "sources"
    manifest = json.loads((args.data_root / "collection_manifest.json").read_text())
    symbols = sorted(
        {p.name.split("_")[0] for p in sources.glob("*_1d.perp.parquet")}
        & {p.name.split("_")[0] for p in sources.glob("*_1d.spot.parquet")}
    )
    if len(symbols) < 8:
        raise ValueError("need at least 8 paired names for cross-sectional study")
    files = [sources / f"{s}_1d.{venue}.parquet" for s in symbols for venue in ("perp", "spot")]
    hashes = {p.name: _sha256(p) for p in files}
    for name, digest in hashes.items():
        if manifest["files"][name]["sha256"] != digest:
            raise ValueError(f"manifest hash mismatch: {name}")
    perp = pl.concat([pl.read_parquet(p) for p in files if ".perp." in p.name])
    spot = pl.concat([pl.read_parquet(p) for p in files if ".spot." in p.name])
    if set(perp["source"].unique().to_list()) != {"binance_usdtm_perp"}:
        raise ValueError("unexpected perp source")
    if set(spot["source"].unique().to_list()) != {"binance_public_data"}:
        raise ValueError("unexpected spot source")
    keys = ["security_id", "event_time"]
    panel = (
        perp.select(keys + ["open", "close", "available_time"])
        .rename({"open": "perp_open", "close": "perp_close", "available_time": "perp_known"})
        .join(
            spot.select(keys + ["open", "close", "available_time"]).rename(
                {
                    "open": "spot_open",
                    "close": "spot_close",
                    "available_time": "spot_known",
                }
            ),
            on=keys,
            how="inner",
        )
        .sort(keys)
        .with_columns((pl.col("perp_close") / pl.col("spot_close") - 1.0).alias("basis"))
        .with_columns(
            pl.col("basis").rolling_mean(30, min_samples=30).over("security_id").alias("basis_mu"),
            pl.col("basis").rolling_std(30, min_samples=30).over("security_id").alias("basis_sd"),
            pl.col("perp_open").shift(-1).over("security_id").alias("next_open"),
            pl.col("perp_close").shift(-1).over("security_id").alias("next_close"),
            pl.col("spot_open").shift(-1).over("security_id").alias("next_spot_open"),
            pl.col("spot_close").shift(-1).over("security_id").alias("next_spot_close"),
            pl.col("event_time").shift(-1).over("security_id").alias("next_time"),
        )
        .with_columns(
            (-(pl.col("basis") - pl.col("basis_mu")) / pl.col("basis_sd"))
            .alias("signal"),
            (pl.col("next_close") / pl.col("next_open") - 1.0).alias("future_return"),
            (
                pl.col("next_spot_close") / pl.col("next_spot_open")
                - pl.col("next_close") / pl.col("next_open")
            ).alias("future_hedged_pair_return"),
        )
    )
    dates = sorted(panel["event_time"].unique().to_list())
    cut = dates[int(len(dates) * 0.8)]
    dev = panel.filter(
        (pl.col("event_time") < cut)
        & (pl.col("next_time") < cut)
        & (pl.col("perp_known") < pl.col("next_time"))
        & (pl.col("spot_known") < pl.col("next_time"))
        & pl.col("signal").is_finite()
        & pl.col("future_return").is_finite()
        & pl.col("future_hedged_pair_return").is_finite()
    )
    by_date = []
    hedged_daily = []
    for (date,), group in dev.group_by("event_time", maintain_order=True):
        if group.height < 8:
            continue
        group = group.sort("signal")
        k = max(2, group.height // 4)
        short = group.head(k)["future_return"].mean()
        long = group.tail(k)["future_return"].mean()
        by_date.append((date, float(long - short), group.height))
        rich = group.filter((pl.col("basis") > 0) & (pl.col("signal") < -1.0))
        hedged_daily.append(
            float(rich["future_hedged_pair_return"].mean()) if rich.height else 0.0
        )
    if len(by_date) < 40:
        raise ValueError("too few development dates")
    spreads = np.asarray([row[1] for row in by_date], dtype=float)
    folds = np.array_split(spreads, 4)
    hedged = np.asarray(hedged_daily, dtype=float)
    hedged_folds = np.array_split(hedged, 4)
    receipt = {
        "schema": "basis_reversion_screen.v1",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "evidence_level": "development_pre_cost_diagnostic",
        "research_only": True,
        "live_pnl_claim": False,
        "bar_interval": "1d",
        "signal": "negative 30-bar z-score of same-asset perp/spot closing basis",
        "target": "next-bar perp open-to-close, long top quartile and short bottom quartile",
        "development_cut": str(cut),
        "n_assets": len(symbols),
        "n_dates": len(by_date),
        "mean_daily_gross_spread": float(np.mean(spreads)),
        "median_daily_gross_spread": float(np.median(spreads)),
        "positive_day_fraction": float(np.mean(spreads > 0)),
        "four_chronological_fold_means": [float(np.mean(fold)) for fold in folds],
        "hedged_pair_rule": "long spot/short perp when basis is positive and z-score exceeds 1",
        "hedged_pair_excludes": "funding, commissions, spread, impact, margin costs",
        "hedged_pair_active_date_fraction": float(np.mean(hedged != 0)),
        "hedged_pair_mean_daily_gross_return": float(np.mean(hedged)),
        "hedged_pair_four_fold_means": [float(np.mean(fold)) for fold in hedged_folds],
        "input_hashes": hashes,
        "script_sha256": _sha256(Path(__file__)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: receipt[k] for k in (
        "n_dates", "mean_daily_gross_spread", "positive_day_fraction",
        "four_chronological_fold_means", "hedged_pair_active_date_fraction",
        "hedged_pair_mean_daily_gross_return", "hedged_pair_four_fold_means",
    )}, indent=2))


if __name__ == "__main__":
    main()
