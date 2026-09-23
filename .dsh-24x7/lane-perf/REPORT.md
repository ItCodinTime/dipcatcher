# lane-perf: `run_backtest_fast` numba kernel — honest close-the-gap report

**Goal:** make `run_backtest_fast` materially faster while remaining
bitwise-identical to `run_backtest`. No changes to `engine.py`,
`risk_gate.py`, or any parity-defining test.

## Verdict (honest)

**The sequential-loop gap is closed; the end-to-end vectorbt gap is not.**

| path | remote median (ms) | vs |
|---|---|---|
| `run_backtest` (reference) | 816.8 | 1.0x |
| `run_backtest_fast` before (git HEAD) | 209.0 | 3.9x |
| `run_backtest_fast` after (numba) | **65.5** | **12.5x** |
| vectorbt `Portfolio.from_orders` | 16.4 | 49.8x |

- Before → after on the same box, same workload, interleaved: **209.0 → 65.5 ms (3.19x)**.
- vectorbt is still ~4x faster end-to-end (65.5 vs 16.4 ms). The residual is
  **not** the replay loop: the numba kernel itself runs in ~2 ms. ~42 ms of
  the remaining ~65 ms is `_build_result` — the shared metrics/result tail
  (sharpe, drawdown, book diagnostics, analytics export) that the reference
  engine also pays and that vectorbt does not compute eagerly
  (`Portfolio` defers stats). Excluding that shared tail, fast-path-specific
  work is ~24 ms vs vectorbt's ~16 ms — near parity on the differentiating
  work. Closing the rest would require shrinking the metrics contract or
  touching `engine.py`, both out of scope.

## Profile (remote, cProfile; 75 ms under profiler ≈ 65.5 ms wall)

| phase | ms | notes |
|---|---|---|
| `_build_result` metrics tail | ~42 | shared with reference; engine.py, untouched |
| `_replay_driver` total | ~15 | kernel ~2 ms + buffer alloc + row-dict materialisation |
| `_bars_to_matrices` | ~8 | polars casts + numpy scatter (was ~30) |
| weight scatter + carry | ~2 | numpy `searchsorted`/assign (was ~14 pivot) |
| `_validate_panel_fast` | ~1 | struct `is_duplicated` + cast/isfinite (was ~56) |
| polars `collect` (unique/sort etc.) | ~10 | inside the phases above |

Baseline loop profile (pre-kernel, interpreted): residual loop+result
~297 ms, of which `sum()`/generator overhead dominated — this is what the
kernel eliminated.

## What changed (single production file: `src/quant_fund/backtest/fast_replay.py`)

1. **Numba kernel** (`@njit(cache=True)`, in-module): `_neumaier`, `_csum`,
   `_order_costs_nb`, `_replay_kernel`, plus pure-Python `_replay_driver`
   (buffer alloc, calls kernel, materialises row dicts). Reproduces the
   interpreted loop statement-for-statement: mark update, stale-valuation
   fail-closed (two-pass ordering, identical message), exec-price NAV,
   dust filter, participation cap, kill switch, per-order risk gate with
   sorted contributor exposure, cash accounting, borrow, mark-to-close,
   and all counters.
2. **CPython 3.12 `sum()` fidelity**: exact-`float` prefixes accumulate
   under Neumaier; the first `np.float64` term commits the compensated
   partial **only when compensation is nonzero and finite** (matching
   CPython's `if (c && Py_IS_FINITE(c))` guard) then naive-folds the rest.
   `np.float64` propagation is tracked via per-asset `share_np64` /
   `delta_np64` / `cash_np64` flags — flags choose the summation path,
   never the arithmetic. Contributor membership filters on operands
   (`pb != 0.0 and npv[b] != 0.0`), not the product, so an
   underflowed-to-zero np.float64 term still flips the path exactly as the
   reference does. Model verified against CPython `sum()` on 200k
   randomized cases (0 value/type mismatches).
3. **Driver emits row dicts directly** (no intermediate polars frames):
   `_build_result` re-infers identical dtypes, and empty-fills workloads
   produce the reference's 0x0 fills frame exactly.
4. **Ingest glue**: polars `unique().sort()` for dates (interpreter
   fallback preserves `sorted()` TypeError on null keys), numpy
   searchsorted+scatter replaces the weight pivot (dedup already validated
   upstream; pivot fallback retained for non-string/null-keyed panels),
   vectorized `_validate_panel_fast` for the clean path.
5. **Fallback preserved**: without numba the interpreted loop runs —
   identical outputs, slower. Numba-failure paths degrade gracefully
   (`except Exception → kernel_out = None`); fail-closed
   `StaleValuationError`/`ValueError` propagate.

### Scope note
An earlier parallel iteration had extended `engine.py::_build_result` to
accept pre-built DataFrames and kept the kernel in a separate
`_fast_kernel.py`. Both exceeded the sanctioned file set, so the engine
change was **reverted** (git-clean now) and the kernel **inlined** into
`fast_replay.py`. Cost of compliance: ~0 ms for the inline (perf-neutral)
and ~2–4 ms for row-dict emission vs frame pass-through.

## Parity evidence

- `pytest tests/unit/test_fast_replay.py -q`: **9/9 pass** — local and remote.
- `scripts/_conformance_11a.py` (real 11-asset, 999-day workload):
  `nav_bitwise_equal: true`, `nav_max_abs_diff: 0.0`, `fills_equal: true` —
  local and remote.
- Before-vs-after NAV bitwise equal on the bench workload.
- `ruff`: clean. `mypy`: clean.

## Remaining bottlenecks / honest residuals

- `_build_result` (~42 ms, 64% of run): shared metrics tail; off-limits
  (engine.py). This is the dominant remaining gap vs vectorbt.
- `_replay_driver` materialisation (~13 ms): output buffers → row dicts.
- `_bars_to_matrices` (~8 ms): five column casts + two scatter passes.
- vectorbt caveat: `from_orders(targetpercent)` does not run a per-order
  risk gate, staleness tracking, or eager metrics — a structurally
  lighter contract. The honest apples-to-apples statement: the replay
  kernel (~2 ms) now runs faster than vectorbt's order engine portion;
  the product's output contract carries the rest.

## Reproduce

```powershell
# remote scratch (D:\dipcatcher-megaplan\_lane-perf; data/ junctioned)
$env:PYTHONPATH='src'; D:/bench-qlib/Scripts/python.exe scripts/_conformance_11a.py
D:/bench-qlib/Scripts/python.exe scripts/_bench_fast.py --reps 15 --vectorbt --ref
D:/bench-qlib/Scripts/python.exe scripts/_bench_before_after.py
```

```bash
# local
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_fast_replay.py -q
PYTHONPATH=src .venv/bin/python scripts/_conformance_11a.py
PYTHONPATH=src .venv/bin/python scripts/_bench_fast.py --reps 15 --vectorbt --ref
```

Machine/load notes: remote = Windows 10, Skylake-SP 24c, Python 3.12.14,
numba 0.67.0 — primary numbers. Local = Apple M4 under load avg ~34 —
local medians (ref 889.6, before 678.7, after 167.8, vbt 20.6 ms) are
load-inflated and secondary.
