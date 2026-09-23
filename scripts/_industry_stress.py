"""Industry-grade evidence: determinism + scale stress for the backtest engine.

Measures (all captured to a JSON receipt):
1. Determinism — run_backtest_fast on identical inputs N times; all NAV/fill
   outputs must be byte-identical (sha256 of serialized frames).
2. Scale — the deep 1d bars (~3300 rows/asset x N assets) vs the incumbent
   bench workload (~999 rows); wall time + peak RSS scaling.
3. Reference parity at scale — run_backtest once on the same panel, NAV
   bitwise-equal check.

research-only; no live-PnL claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

try:
    import resource  # Unix only
except ImportError:
    resource = None  # type: ignore[assignment]

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from quant_fund.backtest.engine import run_backtest  # noqa: E402
from quant_fund.backtest.fast_replay import run_backtest_fast  # noqa: E402
from quant_fund.config.models import AppConfig, CostConfig, RiskGateConfig  # noqa: E402


def _bench_cfg() -> AppConfig:
    """Permissive config identical to scripts/incumbent_bench_vectorbt.py — lets
    the panel trade so the run exercises the full fill path (risk gate enabled
    but wide)."""
    return AppConfig(
        costs=CostConfig(
            commission_bps=10.0,
            half_spread_bps=0.0,
            impact_y=0.0,
            bps_per_turnover=0.0,
            borrow_bps_per_year=0.0,
            financing_bps_per_year=0.0,
            frictionless=False,
            participation_limit=1.0,
        ),
        risk_gate=RiskGateConfig(
            max_order_notional=1e12,
            max_gross=100.0,
            max_net=100.0,
            max_name=1.0,
            max_participation=1.0,
            max_predicted_vol=100.0,
            stale_price_bars=3,
            stale_model_hours=1e9,
        ),
    )


def _panel(bars: pl.DataFrame, n_dates_per_asset: int = 20) -> pl.DataFrame:
    """SMA20-gate causal weight panel identical in spirit to the incumbent bench."""
    px = bars.select("security_id", "event_time", "close").sort("security_id", "event_time")
    px = px.with_columns(pl.col("close").rolling_mean(20).over("security_id").alias("sma20"))
    px = px.with_columns((pl.col("close") > pl.col("sma20")).cast(pl.Float64).alias("sig"))
    # equal-weight long-only over the active names, 0.9 buffer
    px = px.with_columns(
        (pl.col("sig") * 0.9 / pl.col("sig").sum().over("event_time").clip(1.0)).alias(
            "target_weight"
        )
    )
    return px.select("event_time", "security_id", "target_weight").drop_nulls()


def _hash_result(res) -> dict[str, str]:
    nav_h = hashlib.sha256(res.equity.write_csv().encode()).hexdigest()
    fill_h = hashlib.sha256(res.fills.write_csv().encode()).hexdigest()
    return {
        "nav_sha256": nav_h,
        "fills_sha256": fill_h,
        "n_nav": res.equity.height,
        "n_fills": res.fills.height,
    }


def _peak_rss_mb() -> float:
    if resource is None:
        try:
            import psutil

            return float(psutil.Process().memory_info().rss)
        except ImportError:
            return float("nan")
    # ru_maxrss: bytes on Linux, KiB on macOS — record raw + the platform note.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference", action="store_true", help="also run reference engine once")
    args = ap.parse_args()

    frames = [pl.read_parquet(p).sort("event_time") for p in args.bars]
    bars = pl.concat(frames).sort(["event_time", "security_id"])
    if "close_total_return" not in bars.columns:
        bars = bars.with_columns(pl.col("close").alias("close_total_return"))
    weights = _panel(bars)
    cfg = _bench_cfg()  # same permissive-gate config as the incumbent bench

    receipt: dict = {
        "script": str(Path(__file__).name),
        "bars_files": [Path(p).name for p in args.bars],
        "bars_rows": bars.height,
        "n_assets": bars["security_id"].n_unique(),
        "n_dates": bars["event_time"].n_unique(),
        "weight_rows": weights.height,
        "research_only": True,
        "live_pnl_claim": False,
    }

    # --- determinism: N identical runs of the fast path
    runs = []
    for k in range(args.reps):
        t0 = time.perf_counter()
        res = run_backtest_fast(bars, weights, cfg)
        dt = time.perf_counter() - t0
        h = _hash_result(res)
        runs.append({"rep": k, "ms": dt * 1000, **h})
    hashes = {(r["nav_sha256"], r["fills_sha256"]) for r in runs}
    receipt["determinism"] = {
        "reps": args.reps,
        "identical": len(hashes) == 1,
        "nav_sha256": runs[0]["nav_sha256"],
        "fills_sha256": runs[0]["fills_sha256"],
        "ms": [r["ms"] for r in runs],
        "ms_median": float(np.median([r["ms"] for r in runs])),
    }
    receipt["peak_rss_raw"] = _peak_rss_mb()

    # --- reference parity at this scale (single run)
    if args.reference:
        t0 = time.perf_counter()
        ref = run_backtest(bars, weights, cfg)
        ref_ms = (time.perf_counter() - t0) * 1000
        ref_h = _hash_result(ref)
        receipt["reference"] = {
            "ms": ref_ms,
            **ref_h,
            "nav_bitwise_equal": ref_h["nav_sha256"] == runs[0]["nav_sha256"],
            "fills_equal": ref_h["fills_sha256"] == runs[0]["fills_sha256"],
        }
    receipt["peak_rss_final"] = _peak_rss_mb()

    args.out.write_text(json.dumps(receipt, indent=2, default=str))
    print(json.dumps(receipt, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
