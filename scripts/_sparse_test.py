import sys

sys.path.insert(0, r"D:\dipcatcher\src")
sys.path.insert(0, r"D:\dipcatcher\scripts")
from pathlib import Path  # noqa: E402

import incumbent_bench_vectorbt as B  # noqa: E402
import polars as pl  # noqa: E402

frames = B.load_panel(
    [
        Path(rf"D:\dipcatcher\data\raw\sources\{s}_1d.parquet")
        for s in ["btcusdt", "ethusdt", "solusdt"]
    ]
)
bars = pl.concat(list(frames.values()))
# SPARSE panel: weight rows only every 5th bar date
w = B.build_weight_panel(frames, B.SMA_WINDOW)
dates = sorted(w["event_time"].unique().to_list())
keep = set(dates[::5])
ws = w.filter(pl.col("event_time").is_in(list(keep)))
print(f"sparse panel: {ws.height} rows over {len(keep)}/{len(dates)} dates")
from quant_fund.config.models import AppConfig, CostConfig, RiskGateConfig  # noqa: E402

cfg = AppConfig(
    costs=CostConfig(
        commission_bps=B.COMMISSION_BPS,
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
from quant_fund.backtest.engine import run_backtest  # noqa: E402
from quant_fund.backtest.fast_replay import run_backtest_fast  # noqa: E402

r_ref = run_backtest(bars, ws, cfg, initial_nav=B.INIT_NAV)
r_fast = run_backtest_fast(bars, ws, cfg, initial_nav=B.INIT_NAV)
e1, e2 = r_ref.equity["nav"].to_list(), r_fast.equity["nav"].to_list()
same = all(a == b for a, b in zip(e1, e2, strict=False)) and len(e1) == len(e2)
print(
    "bitwise identical:",
    same,
    "| ref fills:",
    r_ref.fills.height,
    "fast fills:",
    r_fast.fills.height,
)
if not same:
    for i, (a, b) in enumerate(zip(e1, e2, strict=False)):
        if a != b:
            print("first divergence at row", i, a, b)
            break
