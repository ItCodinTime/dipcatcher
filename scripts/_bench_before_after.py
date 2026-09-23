"""Before/after benchmark: HEAD fast_replay vs current kernel build.

Interleaves ``_fast_replay_before.run_backtest_fast`` (the pre-kernel
interpreted implementation, extracted verbatim from git HEAD) with the
current ``run_backtest_fast`` on the same 11-asset workload.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _bench_fast import load_workload, make_cfg  # noqa: E402

# Load the frozen HEAD copy as an isolated module — its internal imports
# (engine, config, kill_switch) resolve against the live tree, whose
# engine.py is identical to HEAD.
spec = importlib.util.spec_from_file_location(
    "_fast_replay_before", ROOT / "scripts" / "_fast_replay_before.py"
)
before_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(before_mod)

from quant_fund.backtest.fast_replay import run_backtest_fast  # noqa: E402

bars, weights = load_workload()
cfg = make_cfg()

# Parity between before/after implementations on this workload too.
res_b = before_mod.run_backtest_fast(bars, weights, cfg)
res_a = run_backtest_fast(bars, weights, cfg)
nav_b = np.asarray(res_b.equity["nav"].to_list())
nav_a = np.asarray(res_a.equity["nav"].to_list())
parity = bool(np.array_equal(nav_b, nav_a))

REPS = 15
tb, ta = [], []
run_backtest_fast(bars, weights, cfg)  # warm the numba cache path
before_mod.run_backtest_fast(bars, weights, cfg)
for _ in range(REPS):
    t0 = time.perf_counter()
    before_mod.run_backtest_fast(bars, weights, cfg)
    tb.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    run_backtest_fast(bars, weights, cfg)
    ta.append(time.perf_counter() - t0)

print(json.dumps({
    "before_ms_median": float(np.median(tb) * 1e3),
    "after_ms_median": float(np.median(ta) * 1e3),
    "before_all_ms": [round(t * 1e3, 1) for t in tb],
    "after_all_ms": [round(t * 1e3, 1) for t in ta],
    "before_after_nav_bitwise_equal": parity,
}, indent=2))
