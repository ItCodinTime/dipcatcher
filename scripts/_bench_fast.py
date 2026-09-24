"""Local benchmark + profile harness for run_backtest_fast (lane-perf).

Usage:
  .venv/bin/python scripts/_bench_fast.py [--reps N] [--profile] [--vectorbt]
"""

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

INIT_NAV = 1_000_000.0
SMA_WINDOW = 20
SIDS = [
    "adausdt",
    "avaxusdt",
    "bnbusdt",
    "btcusdt",
    "dogeusdt",
    "ethusdt",
    "linkusdt",
    "ltcusdt",
    "solusdt",
    "trxusdt",
    "xrpusdt",
]


def load_workload(sids=SIDS):
    frames = {}
    for s in sids:
        fr = pl.read_parquet(ROOT / f"data/raw/sources/{s}_1d.parquet").sort("event_time")
        if "close_total_return" not in fr.columns:
            fr = fr.with_columns(pl.col("close").alias("close_total_return"))
        frames[fr["security_id"][0]] = fr
    closes = {sid: f["close"].to_numpy() for sid, f in frames.items()}
    times = next(iter(frames.values()))["event_time"].to_list()
    rows = []
    n = len(frames)
    for i in range(len(times)):
        if i + 1 >= SMA_WINDOW:
            gated = [
                s for s, c in closes.items() if c[i] > float(np.mean(c[i + 1 - SMA_WINDOW : i + 1]))
            ]
            for sid in closes:
                rows.append(
                    {
                        "event_time": times[i],
                        "security_id": sid,
                        "target_weight": 0.9 / n if sid in gated else 0.0,
                    }
                )
    weights = pl.DataFrame(
        rows,
        schema={
            "event_time": pl.Datetime("us", "UTC"),
            "security_id": pl.String,
            "target_weight": pl.Float64,
        },
    )
    return pl.concat(list(frames.values())), weights


def make_cfg():
    from quant_fund.config.models import AppConfig, CostConfig, RiskGateConfig

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


def run_fast(bars, weights, cfg):
    from quant_fund.backtest.fast_replay import run_backtest_fast

    return run_backtest_fast(bars, weights, cfg, initial_nav=INIT_NAV)


def run_ref(bars, weights, cfg):
    from quant_fund.backtest.engine import run_backtest

    return run_backtest(bars, weights, cfg, initial_nav=INIT_NAV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--vectorbt", action="store_true")
    ap.add_argument("--ref", action="store_true")
    args = ap.parse_args()

    bars, weights = load_workload()
    cfg = make_cfg()

    if args.profile:
        run_fast(bars, weights, cfg)  # warmup
        pr = cProfile.Profile()
        pr.enable()
        run_fast(bars, weights, cfg)
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(25)
        print(s.getvalue()[:6000])
        pstats.Stats(pr, stream=s).sort_stats("cumtime").print_stats(15)
        print(s.getvalue()[:3000])
        return

    # interleaved timing
    ts_fast, ts_vb, ts_ref = [], [], []
    vb_args = None
    if args.vectorbt:
        import pandas as pd

        frames = {
            sid: bars.filter(pl.col("security_id") == sid).sort("event_time")
            for sid in sorted(bars["security_id"].unique().to_list())
        }
        sids = sorted(frames)
        idx = pd.DatetimeIndex(pd.to_datetime(frames[sids[0]]["event_time"].to_list()))
        open_px = pd.DataFrame({s: frames[s]["open"].to_numpy() for s in sids}, index=idx)
        close_px = pd.DataFrame({s: frames[s]["close"].to_numpy() for s in sids}, index=idx)
        wmat = (
            weights.to_pandas()
            .assign(event_time=lambda d: pd.to_datetime(d["event_time"]))
            .pivot(index="event_time", columns="security_id", values="target_weight")
            .reindex(index=idx, columns=sids)
            .fillna(0.0)
        )
        vb_args = (open_px, close_px, wmat)

        def run_vb():
            import vectorbt as vbt

            size = wmat.shift(1).fillna(0.0)
            return vbt.Portfolio.from_orders(
                close=close_px,
                size=size,
                size_type="targetpercent",
                price=open_px,
                fees=10.0 / 1e4,
                init_cash=INIT_NAV,
                cash_sharing=True,
                direction="longonly",
                freq="1D",
                group_by=True,
            )

        run_vb()

    run_fast(bars, weights, cfg)  # warmup
    for _ in range(args.reps):
        t0 = time.perf_counter()
        run_fast(bars, weights, cfg)
        ts_fast.append(time.perf_counter() - t0)
        if vb_args is not None:
            t0 = time.perf_counter()
            run_vb()
            ts_vb.append(time.perf_counter() - t0)
        if args.ref:
            t0 = time.perf_counter()
            run_ref(bars, weights, cfg)
            ts_ref.append(time.perf_counter() - t0)

    out = {
        "fast_ms_median": float(np.median(ts_fast) * 1e3),
        "fast_all_ms": [round(t * 1e3, 1) for t in ts_fast],
    }
    if ts_vb:
        out["vectorbt_ms_median"] = float(np.median(ts_vb) * 1e3)
        out["vectorbt_all_ms"] = [round(t * 1e3, 1) for t in ts_vb]
    if ts_ref:
        out["ref_ms_median"] = float(np.median(ts_ref) * 1e3)
    import json

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
