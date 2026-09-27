import cProfile
import io
import pstats
import sys
from pathlib import Path

sys.path.insert(0, r"D:\dipcatcher\src")
sys.path.insert(0, r"D:\dipcatcher\scripts")
import incumbent_bench_vectorbt as B
import polars as pl

bars_paths = [
    Path(r"D:\dipcatcher\data\raw\sources") / f"{s}_1d.parquet"
    for s in ("btcusdt", "ethusdt", "solusdt")
]
frames = B.load_panel(bars_paths)
weights = B.build_weight_panel(frames, B.SMA_WINDOW)
bars = pl.concat(list(frames.values()))
B.run_dipcatcher(bars, weights, B.COMMISSION_BPS, engine="fast")  # warmup
pr = cProfile.Profile()
pr.enable()
B.run_dipcatcher(bars, weights, B.COMMISSION_BPS, engine="fast")
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(15)
print(s.getvalue()[:3500])
