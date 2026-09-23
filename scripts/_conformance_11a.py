"""Bitwise conformance: run_backtest vs run_backtest_fast on the real 11-asset
workload, plus latency medians for both. Remote-only diagnostic."""
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from quant_fund.backtest.engine import run_backtest  # noqa: E402
from quant_fund.backtest.fast_replay import run_backtest_fast  # noqa: E402
from quant_fund.config.models import AppConfig, CostConfig, RiskGateConfig  # noqa: E402

INIT_NAV = 1_000_000.0
SMA_WINDOW = 20
SIDS = ["adausdt", "avaxusdt", "bnbusdt", "btcusdt", "dogeusdt", "ethusdt",
        "linkusdt", "ltcusdt", "solusdt", "trxusdt", "xrpusdt"]

frames = {}
for s in SIDS:
    fr = pl.read_parquet(ROOT / f"data/raw/sources/{s}_1d.parquet").sort("event_time")
    if "close_total_return" not in fr.columns:
        fr = fr.with_columns(pl.col("close").alias("close_total_return"))
    frames[fr["security_id"][0]] = fr

# same causal weight panel as the incumbent bench
closes = {sid: f["close"].to_numpy() for sid, f in frames.items()}
times = next(iter(frames.values()))["event_time"].to_list()
rows = []
n = len(frames)
for i in range(len(times)):
    if i + 1 >= SMA_WINDOW:
        gated = [s for s, c in closes.items() if c[i] > float(np.mean(c[i + 1 - SMA_WINDOW: i + 1]))]
        for sid in gated:
            rows.append({"event_time": times[i], "security_id": sid,
                         "target_weight": 0.9 / n})
weights = pl.DataFrame(rows, schema={"event_time": pl.Datetime("us", "UTC"),
                                     "security_id": pl.String,
                                     "target_weight": pl.Float64})
bars = pl.concat(list(frames.values()))

cfg = AppConfig(
    costs=CostConfig(commission_bps=10.0, half_spread_bps=0.0, impact_y=0.0,
                     bps_per_turnover=0.0, borrow_bps_per_year=0.0,
                     frictionless=False, participation_limit=1.0),
    risk_gate=RiskGateConfig(max_order_notional=1e12, max_gross=100.0,
                             max_net=100.0, max_name=1.0,
                             max_participation=1.0, max_predicted_vol=100.0,
                             stale_price_bars=3, stale_model_hours=1e9),
)

ref = run_backtest(bars, weights, cfg, initial_nav=INIT_NAV)
fast = run_backtest_fast(bars, weights, cfg, initial_nav=INIT_NAV)

nav_r = np.asarray(ref.equity["nav"].to_list())
nav_f = np.asarray(fast.equity["nav"].to_list())
bit_eq_nav = bool(np.array_equal(nav_r, nav_f))

fill_eq = True
cols_diff = []
if ref.fills.height != fast.fills.height:
    fill_eq = False
else:
    for c in ref.fills.columns:
        a, b = ref.fills[c].to_list(), fast.fills[c].to_list()
        if c in ("fill_time", "signal_time", "security_id"):
            ok = a == b
        else:
            ok = np.array_equal(np.asarray(a, dtype=object), np.asarray(b, dtype=object))
        if not ok:
            fill_eq = False
            cols_diff.append(c)

ts_ref, ts_fast = [], []
for _ in range(6):
    t0 = time.perf_counter()
    run_backtest(bars, weights, cfg, initial_nav=INIT_NAV)
    ts_ref.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    run_backtest_fast(bars, weights, cfg, initial_nav=INIT_NAV)
    ts_fast.append(time.perf_counter() - t0)

out = {
    "nav_bitwise_equal": bit_eq_nav,
    "nav_max_abs_diff": float(np.abs(nav_r - nav_f).max()) if not bit_eq_nav else 0.0,
    "fills_equal": fill_eq,
    "fills_diff_cols": cols_diff,
    "ref_ms_median": float(np.median(ts_ref) * 1e3),
    "fast_ms_median": float(np.median(ts_fast) * 1e3),
    "ref_all_ms": [round(t * 1e3, 1) for t in ts_ref],
    "fast_all_ms": [round(t * 1e3, 1) for t in ts_fast],
}
print(json.dumps(out, indent=2))
