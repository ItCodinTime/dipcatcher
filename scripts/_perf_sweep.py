"""P4.1/P6.8 perf sweep harness — SYNTHETIC 11-asset multi-year daily panel.

Profiles ``quant_fund.backtest.engine.run_backtest`` on the same workload
shape as ``scripts/_bench_fast.py`` (SMA-20 gated target weights, 11 names)
but built from ``tests.perf.synthetic.daily_ohlcv`` instead of the raw
parquets, so every number here is a SYNTHETIC throughput measurement, not
market evidence.

Usage:
    uv run python scripts/_perf_sweep.py [--reps N] [--days N] [--stats-dir DIR]

Outputs per engine path:
  * wall-clock median (min/max) over --reps runs
  * cProfile top-20 by cumulative and by internal (tottime) time
  * tracemalloc top-20 allocation sites (event loop only)
"""

import argparse
import cProfile
import io
import pstats
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tests.perf.synthetic import daily_ohlcv  # noqa: E402

from quant_fund.backtest.engine import _run_backtest_event_loop, run_backtest  # noqa: E402
from quant_fund.config.models import AppConfig, CostConfig, RiskGateConfig  # noqa: E402

INIT_NAV = 1_000_000.0
SMA_WINDOW = 20
N_ASSETS = 11
N_DAYS = 1000  # ~4 years of weekday bars


def build_workload(
    n_assets: int = N_ASSETS, n_days: int = N_DAYS
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """SYNTHETIC bars + SMA-gated target weights (mirrors _bench_fast.load_workload)."""
    bars = daily_ohlcv(n_assets, n_days).with_columns(
        (pl.col("close") * pl.col("volume")).alias("adv"),
        pl.lit(0.02).alias("vol_20"),
    )
    pivot = bars.select("security_id", "event_time", "close").pivot(
        on="security_id", index="event_time", values="close"
    )
    sids = sorted(bars["security_id"].unique().to_list())
    times = pivot["event_time"].to_list()
    rows = []
    n = len(sids)
    closes = {sid: pivot[sid].to_numpy() for sid in sids}
    for i in range(len(times)):
        if i + 1 >= SMA_WINDOW:
            gated = {
                s for s, c in closes.items() if c[i] > float(np.mean(c[i + 1 - SMA_WINDOW : i + 1]))
            }
            for sid in sids:
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
    return bars, weights


def make_cfg() -> AppConfig:
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


def _time(fn, bars, weights, cfg, reps: int) -> dict[str, float]:
    fn(bars, weights, cfg, initial_nav=INIT_NAV)  # warmup
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(bars, weights, cfg, initial_nav=INIT_NAV)
        samples.append(time.perf_counter() - t0)
    return {
        "median_s": float(np.median(samples)),
        "min_s": float(np.min(samples)),
        "max_s": float(np.max(samples)),
        "reps": reps,
    }


def _profile(fn, bars, weights, cfg) -> pstats.Stats:
    pr = cProfile.Profile()
    pr.enable()
    fn(bars, weights, cfg, initial_nav=INIT_NAV)
    pr.disable()
    return pstats.Stats(pr)


def _top(stats: pstats.Stats, sort: str, n: int = 20) -> str:
    s = io.StringIO()
    stats.stream = s
    stats.sort_stats(sort).print_stats(n)
    return s.getvalue()


def _alloc_trace(fn, bars, weights, cfg, n: int = 20) -> str:
    tracemalloc.start()
    fn(bars, weights, cfg, initial_nav=INIT_NAV)
    snap = tracemalloc.take_snapshot()
    tracemalloc.stop()
    out = io.StringIO()
    for i, stat in enumerate(snap.statistics("lineno")[:n], 1):
        out.write(f"{i:2d}. {stat}\n")
    return out.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--days", type=int, default=N_DAYS)
    ap.add_argument("--assets", type=int, default=N_ASSETS)
    ap.add_argument("--stats-dir", type=Path, default=None)
    args = ap.parse_args()

    bars, weights = build_workload(args.assets, args.days)
    cfg = make_cfg()
    print(
        f"workload: SYNTHETIC {args.assets} assets x {args.days} days "
        f"({bars.height} bars, {weights.height} weight rows)"
    )

    paths = {
        "auto_dispatch": lambda b, w, c, **kw: run_backtest(b, w, c, **kw),
        "event_loop": lambda b, w, c, **kw: _run_backtest_event_loop(b, w, c, **kw),
        "fast_pinned": lambda b, w, c, **kw: run_backtest(b, w, c, fast=True, **kw),
    }
    for name, fn in paths.items():
        t = _time(fn, bars, weights, cfg, args.reps)
        print(
            f"{name}: median {t['median_s'] * 1e3:.1f} ms "
            f"(min {t['min_s'] * 1e3:.1f}, max {t['max_s'] * 1e3:.1f}, n={t['reps']})"
        )
        stats = _profile(fn, bars, weights, cfg)
        if args.stats_dir:
            args.stats_dir.mkdir(parents=True, exist_ok=True)
            stats.dump_stats(str(args.stats_dir / f"{name}.pstats"))
        print(f"== {name}: top-20 cumulative ==")
        print(_top(stats, "cumulative"))
        print(f"== {name}: top-20 tottime ==")
        print(_top(stats, "tottime"))

    print("== event_loop: tracemalloc top-20 ==")
    print(_alloc_trace(paths["event_loop"], bars, weights, cfg))


if __name__ == "__main__":
    main()
