"""Profile run_backtest_fast on the 11-asset workload."""
import cProfile
import io
import pstats
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from quant_fund.backtest.fast_replay import run_backtest_fast  # noqa: E402
from quant_fund.config.models import AppConfig, CostConfig, RiskGateConfig  # noqa: E402

SIDS = ["adausdt", "avaxusdt", "bnbusdt", "btcusdt", "dogeusdt", "ethusdt",
        "linkusdt", "ltcusdt", "solusdt", "trxusdt", "xrpusdt"]
frames = {}
for s in SIDS:
    fr = pl.read_parquet(ROOT / f"data/raw/sources/{s}_1d.parquet").sort("event_time")
    if "close_total_return" not in fr.columns:
        fr = fr.with_columns(pl.col("close").alias("close_total_return"))
    frames[fr["security_id"][0]] = fr
closes = {sid: f["close"].to_numpy() for sid, f in frames.items()}
times = next(iter(frames.values()))["event_time"].to_list()
rows = []
n = len(frames)
for i in range(len(times)):
    if i + 1 >= 20:
        gated = [s for s, c in closes.items() if c[i] > float(np.mean(c[i + 1 - 20: i + 1]))]
        for sid in gated:
            rows.append({"event_time": times[i], "security_id": sid, "target_weight": 0.9 / n})
weights = pl.DataFrame(rows, schema={"event_time": pl.Datetime("us", "UTC"),
                                     "security_id": pl.String, "target_weight": pl.Float64})
bars = pl.concat(list(frames.values()))
cfg = AppConfig(
    costs=CostConfig(commission_bps=10.0, participation_limit=1.0),
    risk_gate=RiskGateConfig(max_order_notional=1e12, max_gross=100.0,
                             max_net=100.0, max_name=1.0, max_participation=1.0,
                             max_predicted_vol=100.0, stale_price_bars=3,
                             stale_model_hours=1e9),
)
run_backtest_fast(bars, weights, cfg)  # warmup
pr = cProfile.Profile()
pr.enable()
run_backtest_fast(bars, weights, cfg)
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
print(s.getvalue())
