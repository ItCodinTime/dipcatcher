"""Run the Dip Quality Score bench over a directory of 1d bar parquets.

Produces an honest research receipt: event counts, the unconditional
recovery baseline per horizon, and proper scores for the climatology
forecaster (baseline frequency as the probability — in-sample, labeled as
such). Real data, research-scoped, never a live-performance claim.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from fx1.bench.dip import (
    DipForecast,
    assert_bench_output_honest,
    detect_dip_events,
    evaluate_forecasts,
    unconditional_baseline,
)

BENCH_DISCLAIMER = (
    "Research/backtest evidence on historical data, not live performance. "
    "The climatology forecaster is an in-sample descriptive baseline; it is "
    "not a tradeable signal and authorizes nothing."
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_dip_bench(
    data_dir: str | Path,
    *,
    threshold: float = 0.10,
    horizons_bars: dict[str, int] | None = None,
    glob_pattern: str = "*_1d.parquet",
) -> dict:
    """Detect dips across every series in *data_dir* and score climatology."""
    import polars as pl

    root = Path(data_dir)
    files = sorted(root.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"no {glob_pattern} series under {root}")

    all_events = []
    per_asset: dict[str, dict[str, int]] = {}
    inputs: dict[str, str] = {}
    for path in files:
        df = pl.read_parquet(path).sort("event_time")
        asset = df["symbol"][0] if "symbol" in df.columns else path.stem
        closes = df["close"].to_list()
        dates = [str(t)[:10] for t in df["event_time"].to_list()]
        events = detect_dip_events(
            closes,
            dates,
            str(asset),
            threshold=threshold,
            horizons_bars=horizons_bars,
        )
        all_events.extend(events)
        per_asset[str(asset)] = {"bars": len(closes), "events": len(events)}
        inputs[path.name] = _file_sha256(path)

    baseline = unconditional_baseline(all_events)
    # Climatology forecaster: predict the unconditional recovery frequency.
    forecasts = [
        DipForecast(
            asset=e.asset,
            trough_date=e.trough_date,
            probabilities={h: p for h, p in baseline.items() if p == p},
        )
        for e in all_events
    ]
    metrics = evaluate_forecasts(all_events, forecasts)
    assert_bench_output_honest(metrics)

    return {
        "schema": "fx1.dip_bench/v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "research_only": True,
        "live_pnl_claim": False,
        "disclaimer": BENCH_DISCLAIMER,
        "data": "historical 1d bars (real market data; descriptive research)",
        "params": {
            "threshold": threshold,
            "horizons_bars": horizons_bars or {"1m": 21, "3m": 63, "6m": 126, "12m": 252},
        },
        "inputs_sha256": inputs,
        "per_asset": per_asset,
        "n_events": len(all_events),
        "baseline_recovery": baseline,
        "climatology_forecast": {
            "definition": "P(recovery) = unconditional baseline frequency "
            "(in-sample, descriptive — not a signal)",
            "metrics": metrics,
        },
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Dip Quality Score bench run")
    parser.add_argument("--data-dir", default="data/raw/sources")
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=0.10)
    args = parser.parse_args(argv)
    receipt = run_dip_bench(args.data_dir, threshold=args.threshold)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"events": receipt["n_events"], "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
