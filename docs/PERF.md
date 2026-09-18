# Performance notes (Phase 18)

Honesty: timings below are **SYNTHETIC / lab** micro-benchmarks on a laptop.
They measure infrastructure throughput, not live trading edge.
`live_pnl_claim=false`; all figures are research-only.

## Hot paths

| Path | Role |
|------|------|
| `pipeline.dataset.panel` | Gold feature/label join used by every `forecast_asof` |
| `pipeline.forecast.optimize_asof` | Mean-variance weights as-of a decision date |
| `pipeline.forecast.build_causal_weight_panel` | Causal loop calling `optimize_asof` per date |
| `research.benches` | Research notebook family scores |

## Optimizations landed

1. **Process-local gold panel cache** (`dataset._PANEL_CACHE` / `clear_panel_cache`).
   Keys are bound to canonical content digests plus relevant source metadata
   (path, size, and mtime); a matching metadata guard is required before reusing a
   digest. This avoids stale reuse when a file is rewritten while preserving fast
   hot reads. PIT validation still runs on cache miss.
2. **Ranker joblib cache** (`forecast._RANKER_CACHE`) — avoid re-loading on every asof.
   Stale feature-count mismatch falls back to transparent momentum scores (no crash).
3. **Causal panel reuses one shared frame** — `build_causal_weight_panel` loads panel + ranker once and passes `frame=` into each `optimize_asof`.
4. **Conformal result cache** (`forecast._CONFORMAL_CACHE`) — keyed by `(asof, alpha, height, columns, row_hash)`.
5. **Wrappee fit cache** (`forecast._WRAPPEE_CACHE` / `clear_wrappee_cache` / `wrappee_cal_fingerprint`) —
   reuses scaled Student-`t` / Gaussian fits when the train identity and selected family match.
   Production keys include train/cal date keys, alpha, label column, sizes, train moments, and a
   deterministic SHA-256 digest of the train target/scale arrays, so same-moment data cannot reuse
   a stale fit. The cache is bounded and LRU-evicted; hits refresh recency instead of clearing all
   entries at capacity. Dominant cost inside `conformal_sets_asof` is Student-`t` MLE; cache hits
   avoid that. Family selection remains fresh on each calibration window, while train-fit reuse is
   safe across calibration slides when the train content is unchanged.
6. **Research-only interval skip** (`fusion.skip_intervals=true`) — `forecast_asof` skips
   `conformal_sets_asof` for weight-only smoke benches. Caps are absent; not a live claim path.

Turnover causality (`w_prev`) keeps the date loop sequential; date-parallelism is deferred until a process-safe split exists without breaking TC.

## Measured timings (this machine, research.yaml SYNTHETIC)

### Wave 5 rebench (`data/metadata/perf_bench.json`)

```json
{
  "label": "SYNTHETIC research micro-bench \u2014 not live P&L",
  "wave": 5,
  "n_sample_dates": 25,
  "panel_cold_min_s": 0.024424,
  "panel_hot_min_s": 9.2e-05,
  "panel_speedup_x": 266.2,
  "optimize_asof_cold_min_s": 0.217292,
  "optimize_asof_hot_min_s": 0.012461,
  "optimize_asof_speedup_x": 17.44,
  "wrappee_conformal_cold_min_s": 0.132526,
  "wrappee_conformal_hot_same_asof_min_s": 0.023554,
  "wrappee_speedup_same_asof_x": 5.63,
  "wrappee_cache_hit_on_adjacent": true,
  "causal_panel_25d_s": 2.841625,
  "causal_rows": 885,
  "live_pnl_claim": false,
  "research_only": true
}
```

### Wave 2–5 summary (SYNTHETIC — not live P&L)

| Metric | Wave 2 | Wave 3 | Wave 4 | Wave 5 |
|--------|--------|--------|--------|--------|
| Panel cold→hot | ~292× | ~281× | ~328× | **~266×** |
| optimize_asof cold→hot | ~16.5× | ~13.5× | ~17.3× | **~17.4× (0.217→0.012s)** |
| Wrappee-only same-asof | (landed) | ~6.0× | ~5.5× | **~5.6×** |
| Adjacent wrappee hit | — | false | false | **true** (train-primary) |
| Causal 25d | ~3.32s | ~3.64s | ~3.72s | **~2.84s** (885 rows) |

Interpretation:

- Panel cold → hot speedup ≈ **266×** (parquet+PIT+join avoided; absolute times jitter).
- `optimize_asof` cold → hot ≈ **17.4×**.
- Wrappee-only same-asof ≈ **5.6×**.
- **Adjacent-date wrappee hit is now true** when the 70% train cut holds across a
  one-day hist growth (train-primary fingerprint; cal slide alone does not
  invalidate). Label/alpha/config changes still miss (stale-reuse tests).
- Causal 25-date panel ≈ **2.84s** end-to-end (885 rows).

### Wave 5 engineering (not vendor/live)

- Train-primary wrappee fingerprint + adjacent reuse docs + stale-reuse tests.
- Paper multi-challenger + rolling L1; richer promotion dry-run JSON (schema v2).
- Mid-run kill switch E2E + risk_gate reject accounting (separate from kill/total).
- Research CLI `DATA_LABEL` / BH-FDR families helpers + regression tests.
- Full `pytest -m "not network"` green.

## Wave 7 engineering (not vendor/live)

- Wrappee family **re-select on enlarged cal** with train-fit reuse
  (`select_wrappee_family_name` + `fit_scaled_wrappee` +
  `resolve_wrappee_reselect_cached` / `wrappee_fit_cache_key`).
  Selection always fresh on current cal; train MLE reused when
  `(train-primary fp, family)` hits. Alpha/label still invalidate.
- Attribution hook fixtures + fail-closed empty/mismatch inputs.
- Paper `analytics_export.json` schema validation (`validate_analytics_export`)
  fail-closed on `live_pnl_claim=true`.
- **Skipped**: batched polars asof filter micro-opt — not clearly
  correctness-preserving without a shared date-index plan; remains a next lever.

### Wave 8 engineering notes (no timing rebench)

- **Date-index / day-slice helper landed** (`build_event_time_day_index` +
  `slice_day` in `pipeline/forecast.py`). Exact `event_time == asof` day
  extracts are correctness-preserving (iso key via `_date_keys`); wired through
  `forecast_asof` / `conformal_sets_asof` / `optimize_asof` and built once in
  `build_causal_weight_panel`. Cumulative `event_time <= asof` history filters
  intentionally **not** replaced (PIT/causality clarity).
- No timing rebench this wave (helper is correctness + micro structure; absolute
  causal wall time not remeasured overnight).
- Christoffersen CC extremes, PSR/DSR Bailey–LdP fixtures, ReplayClock identity,
  risk_gate reject accounting properties — tests only (no hot-path change).


### Wave 9 engineering notes (no timing rebench)

- **`history_upto(asof)` helper landed** (`pipeline/forecast.py`) — concat of
  day-index slices with **datetime** `<=` compares (not ISO string order).
  Strict equivalence tests vs `filter(event_time <= asof)` for sorted UTC and
  naive panels, duplicate exact timestamps, and empty-before-all. Mixed
  naive/aware rejected by Polars at frame build.
- **Production wiring (superseded by Wave 10)**: Wave 9 deferred wiring due to
  unsorted reorder risk; Wave 10 landed sort-contract gate + day_index wiring.
- Concentration / executable_alpha / portfolio_conformal extremes /
  promotion_dry_run fail-closed — tests + schema only (no hot-path change).

### Wave 10 engineering notes (history wiring landed)

- **Order-preserving `history_upto` path**: `HISTORY_SORT_KEYS=(event_time,
  security_id)`, `under_history_sort_contract`, `sort_for_history`. Production
  panels already sort by these keys (`dataset.py`).
- **Wiring landed**: `history_for_calibration(..., day_index=)` uses
  `history_upto` **only** when the sort contract holds; unsorted callers fall
  back to explicit `<=` filter (no silent reorder). `conformal_sets_asof`
  passes its `day_index` through.
- Equivalence tests: sorted filter ↔ day-index; unsorted+index → filter
  fallback; sort-then-index matches filter order.
- No timing rebench this wave (contract check is cheap vs Student-t MLE;
  absolute causal wall not remeasured overnight).

### Wave 11 rebench (`data/metadata/perf_bench.json`)

Post–Wave-10 order-preserving `history_upto` / sort-contract wiring.
Honesty: **SYNTHETIC / lab** micro-bench; `live_pnl_claim=false`; not live P&L.

```json
{
  "label": "SYNTHETIC research micro-bench \u2014 not live P&L",
  "wave": 11,
  "n_sample_dates": 25,
  "panel_cold_min_s": 0.0214,
  "panel_hot_min_s": 4.7e-05,
  "panel_speedup_x": 456.54,
  "optimize_asof_cold_min_s": 0.213602,
  "optimize_asof_hot_min_s": 0.011212,
  "optimize_asof_speedup_x": 19.05,
  "wrappee_conformal_cold_min_s": 0.13577,
  "wrappee_conformal_hot_same_asof_min_s": 0.0847,
  "wrappee_speedup_same_asof_x": 1.6,
  "conformal_full_cache_hit_same_asof_min_s": 0.001793,
  "wrappee_cache_hit_on_adjacent": true,
  "wrappee_adjacent_call_s": 0.084761,
  "wrappee_cache_size_after_adjacent": 1,
  "causal_panel_25d_s": 4.325052,
  "causal_panel_25d_all_s": [
    4.325052,
    4.340103
  ],
  "causal_rows": 885,
  "live_pnl_claim": false,
  "research_only": true,
  "history_upto_wiring": true,
  "note": "Wave 11 rebench after order-preserving history_upto wiring. wrappee_* same-asof clears conformal cache only (wrappee warm); conformal_full_cache_hit_* is full result-cache hit."
}
```

Takeaways (research-only):

- Panel cold → hot ≈ **456.54×** (0.0214s → 4.7e-05s).
- `optimize_asof` cold → hot ≈ **19.05×** (0.213602s → 0.011212s).
- Wrappee-only same-asof (conformal cache cleared, wrappee warm) ≈ **1.6×**
  (0.13577s → 0.0847s).
  Weaker than Wave 5 (~5.6×) on this path: residual conformal work still dominates;
  wrappee MLE reuse alone is not the whole story after history wiring.
- Full conformal result-cache same-asof hit ≈ **0.001793s** (near-free).
- Adjacent-date wrappee hit **true** (cache size stayed 1; adjacent call ≈ 0.084761s).
- Causal 25-date panel ≈ **4.325052s** (885 rows) — slower than Wave 5 (~2.84s) on this machine;
  do **not** treat as live edge. Likely contributors: process noise + `history_upto`/sort-contract path on every asof. No parallel causal dates (w_prev still sequential).

### Wave 11 engineering notes

- PERF rebench after Wave 10 `history_upto` production wiring (honest timings above).
- Regime extremes fixtures: GaussianHMM probs sum; SingleState; VolThreshold boundaries;
  invalid `n_states` fail-closed (`GaussianHMMRegime` raises `ValueError`).
- `factor_cov` closed-form / PSD + DCC edge fixtures (`test_covariance.py`).
- Paper `cash_ledger.parquet` schema validation in `validate_ledger_schema` (required columns +
  finite `cash_delta`); roundtrip + fail-closed tests.


### Wave 12 rebench (`data/metadata/perf_bench.json`)

Causal wall recovery after Wave 10–11 `history_upto` day-index concat regression.
Honesty: **SYNTHETIC / lab** micro-bench; `live_pnl_claim=false`; not live P&L.

**Root cause (profiled)**: `history_for_calibration` → `history_upto` day-index
`pl.concat` of day slices was ~**16× slower** than `filter(event_time <= cutoff)`
(~7.8ms vs ~0.5ms per call on this panel). The concat path was only used under the
sort contract — exactly where filter/prefix already preserve order.

**Fix (correctness-preserving)**:
- `history_prefix_upto`: O(log n) Series index binary search + `slice` under
  `HISTORY_SORT_KEYS` contract (~0.023ms vs filter ~0.49ms vs old concat ~7.8ms).
- `history_upto(..., assume_sorted=)` prefers prefix when sorted; unsorted still
  uses chrono concat (may reorder — production never wires that).
- `history_for_calibration(..., assume_sorted=, event_times=)` uses prefix;
  skips per-asof contract re-sort when causal loop already sorted once.
- `build_causal_weight_panel` sorts once if needed, passes `assume_sorted=True`
  + shared `event_times` through `optimize_asof` → `forecast_asof` → conformal.

```json
{
  "label": "SYNTHETIC research micro-bench — not live P&L",
  "wave": 12,
  "n_sample_dates": 25,
  "panel_cold_min_s": 0.023738,
  "panel_hot_min_s": 4.725e-05,
  "panel_speedup_x": 502.39,
  "optimize_asof_cold_min_s": 0.222834,
  "optimize_asof_hot_min_s": 0.011496,
  "optimize_asof_speedup_x": 19.38,
  "wrappee_conformal_cold_min_s": 0.133418,
  "wrappee_conformal_hot_same_asof_min_s": 0.085633,
  "wrappee_speedup_same_asof_x": 1.56,
  "conformal_full_cache_hit_same_asof_min_s": 0.001896,
  "wrappee_cache_hit_on_adjacent": false,
  "causal_panel_25d_s": 3.188822,
  "causal_rows": 885,
  "live_pnl_claim": false,
  "research_only": true,
  "history_prefix_fast_path": true
}
```

Takeaways (research-only):

| Metric | Wave 5 | Wave 11 | **Wave 12** |
|--------|--------|---------|-------------|
| Causal 25d | ~2.84s | ~4.33s | **~3.19s** |
| Panel cold→hot | ~266× | ~457× | **~502×** |
| optimize_asof cold→hot | ~17.4× | ~19.1× | **~19.4×** |
| Wrappee-only same-asof | ~5.6× | ~1.6× | **~1.56×** |
| Adjacent wrappee hit | true | true | **false** (this sample) |

- Causal recovered **~1.14s** vs Wave 11 (4.33→3.19); still **~0.35s** above Wave 5
  on this machine — not claimed fully restored; residual is mostly conformal/MLE
  work added in later waves, not history concat.
- Wrappee-only same-asof ~**1.56×**: expected residual — clearing conformal cache
  still pays Mondrian/CQR + `design_matrix` / row-hash; full conformal hit ≈0.0019s.
  No fake cache hits invented.
- Adjacent wrappee hit **false** on this rebench (cache size 2) — train-primary
  fingerprint can miss when hist grows; Wave 5/11 saw true on other samples.

### Wave 12 engineering notes

- Profiled `history_for_calibration` / `history_upto` / sort-contract; day-index
  concat removed from hot path under sort contract.
- PIT + sort-contract tests extended (`history_prefix_upto` equivalence,
  `assume_sorted` calibration path).
- Exact `slice_day` day-index retained (cheap); cumulative history uses prefix.




### Wave 43 — `assume_sorted` fail-closed (no rebench)

Honesty only: when `assume_sorted=True` but the frame is nonempty with `event_time` and **not** under `HISTORY_SORT_KEYS`, `history_upto` / `history_for_calibration` / optimize trailing-hist raise `ValueError` instead of taking `history_prefix_upto`. Happy path still pays one `under_history_sort_contract` check (already required for honesty). **No PERF rebench** this wave — skip invented wall deltas; production callers that already sort once then pass `assume_sorted=True` are unchanged when the contract holds.

### Wave 19 rebench (`data/metadata/perf_bench.json`)

Honesty: **SYNTHETIC / lab** micro-bench after Waves 12–18 (distribution extremes +
docs only this wave — no intentional hot-path change). `live_pnl_claim=false`;
not live P&L.

```json
{
  "label": "SYNTHETIC research micro-bench — not live P&L",
  "wave": 19,
  "n_sample_dates": 25,
  "panel_cold_min_s": 0.024335,
  "panel_hot_min_s": 4.4e-05,
  "panel_speedup_x": 553.07,
  "optimize_asof_cold_min_s": 0.205258,
  "optimize_asof_hot_min_s": 0.010698,
  "optimize_asof_speedup_x": 19.19,
  "wrappee_conformal_cold_min_s": 0.130016,
  "wrappee_conformal_hot_same_asof_min_s": 0.081685,
  "wrappee_speedup_same_asof_x": 1.59,
  "conformal_full_cache_hit_same_asof_min_s": 0.001671,
  "wrappee_cache_hit_on_adjacent": false,
  "causal_panel_25d_s": 3.145465,
  "causal_rows": 885,
  "live_pnl_claim": false,
  "research_only": true,
  "history_prefix_fast_path": true
}
```

Takeaways (research-only):

| Metric | Wave 5 | Wave 11 | Wave 12 | **Wave 19** |
|--------|--------|---------|---------|-------------|
| Causal 25d | ~2.84s | ~4.33s | ~3.19s | **~3.15s** |
| Panel cold→hot | ~266× | ~457× | ~502× | **~553×** |
| optimize_asof cold→hot | ~17.4× | ~19.1× | ~19.4× | **~19.2×** |
| Wrappee-only same-asof | ~5.6× | ~1.6× | ~1.56× | **~1.59×** |
| Adjacent wrappee hit | true | true | false | **false** |

- Causal ≈ **3.15s** — in the Wave 12 band; still ~0.30s above Wave 5 best (~2.84s)
  on this machine. Not claimed as a new optimization win (no hot-path change).
- Wrappee-only same-asof ~**1.59×** (conformal cleared each rep; wrappee warm) —
  residual Mondrian/CQR/`design_matrix` expected; full conformal hit ≈0.0017s.
- Adjacent wrappee hit **false** (cache size 2) — same honesty as Wave 12 sample.
- Absolute panel/optimize speedups jitter with machine noise; do not treat as edge.

### Wave 19 engineering notes

- Distribution extremes fixtures (`test_distribution_extremes.py`) + fail-closed
  unfitted `predict` on Gaussian / ScaledGaussian / ScaledStudentT / LinearQuantile.
- PERF rebench only (above); no vendor/live/parallel-causal work.


### Wave 28 rebench (`data/metadata/perf_bench.json`)

Honesty: **SYNTHETIC / lab** micro-bench after Waves 25–27 correctness (CPCV
per-group purge, TrialLedger empty DSR NaN, panel/label fail-closed) — no
intentional hot-path change. `live_pnl_claim=false`; not live P&L.

```json
{
  "label": "SYNTHETIC research micro-bench — not live P&L",
  "wave": 28,
  "n_sample_dates": 25,
  "panel_cold_min_s": 0.02248,
  "panel_hot_min_s": 4.433e-05,
  "panel_speedup_x": 507.06,
  "optimize_asof_cold_min_s": 0.201606,
  "optimize_asof_hot_min_s": 0.009862,
  "optimize_asof_speedup_x": 20.44,
  "wrappee_conformal_cold_min_s": 0.128244,
  "wrappee_conformal_hot_same_asof_min_s": 0.080428,
  "wrappee_speedup_same_asof_x": 1.59,
  "conformal_full_cache_hit_same_asof_min_s": 0.001984,
  "wrappee_cache_hit_on_adjacent": false,
  "causal_panel_25d_s": 3.134587,
  "causal_rows": 885,
  "live_pnl_claim": false,
  "research_only": true,
  "history_prefix_fast_path": true
}
```

Takeaways (research-only):

| Metric | Wave 5 | Wave 12 | Wave 19 | **Wave 28** |
|--------|--------|---------|---------|-------------|
| Causal 25d | ~2.84s | ~3.19s | ~3.15s | **~3.13s** |
| Panel cold→hot | ~266× | ~502× | ~553× | **~507×** |
| optimize_asof cold→hot | ~17.4× | ~19.4× | ~19.2× | **~20.4×** |
| Wrappee-only same-asof | ~5.6× | ~1.56× | ~1.59× | **~1.59×** |
| Adjacent wrappee hit | true | false | false | **false** |

- Causal ≈ **3.13s** — Wave 19/12 band; still ~0.29s above Wave 5 best (~2.84s).
  Not claimed as a new optimization win (correctness-only waves since 19).
- Wrappee-only same-asof ~**1.59×** (conformal cleared each rep; wrappee warm) —
  residual Mondrian/CQR/`design_matrix` expected; full conformal hit ≈0.0020s.
- Adjacent wrappee hit **false** (cache size 2) — same honesty as Wave 12/19.
- Absolute panel/optimize speedups jitter with machine noise; do not treat as edge.

### Wave 28 engineering notes

- Panel cache / `design_matrix` fail-closed: missing explicit feature columns,
  empty frame, missing label, all-null rows after drop_nulls.
- `build_labels` empty / missing OHLCV fail-closed (mirrors feature edges).
- PERF rebench only (above); no vendor/live/parallel-causal work.


### Wave 39 rebench (`data/metadata/perf_bench.json`)

Honesty: **SYNTHETIC / lab** micro-bench after Waves 29–38 correctness/edges
(universe/corp-actions/calendars/security_master/scoring/cross_section, etc.) —
no intentional hot-path change. `live_pnl_claim=false`; not live P&L.

```json
{
  "label": "SYNTHETIC research micro-bench — not live P&L",
  "wave": 39,
  "n_sample_dates": 25,
  "panel_cold_min_s": 0.022728,
  "panel_hot_min_s": 4.608e-05,
  "panel_speedup_x": 493.18,
  "optimize_asof_cold_min_s": 0.285616,
  "optimize_asof_hot_min_s": 0.009693,
  "optimize_asof_speedup_x": 29.47,
  "wrappee_conformal_cold_min_s": 0.126406,
  "wrappee_conformal_hot_same_asof_min_s": 0.081675,
  "wrappee_speedup_same_asof_x": 1.55,
  "conformal_full_cache_hit_same_asof_min_s": 0.001801,
  "wrappee_cache_hit_on_adjacent": false,
  "causal_panel_25d_s": 3.14593,
  "causal_rows": 885,
  "live_pnl_claim": false,
  "research_only": true,
  "history_prefix_fast_path": true
}
```

Takeaways (research-only):

| Metric | Wave 5 | Wave 12 | Wave 28 | **Wave 39** |
|--------|--------|---------|---------|-------------|
| Causal 25d | ~2.84s | ~3.19s | ~3.13s | **~3.15s** |
| Panel cold→hot | ~266× | ~502× | ~507× | **~493×** |
| optimize_asof cold→hot | ~17.4× | ~19.4× | ~20.4× | **~29.5×** |
| Wrappee-only same-asof | ~5.6× | ~1.56× | ~1.59× | **~1.55×** |
| Adjacent wrappee hit | true | false | false | **false** |

- Causal ≈ **3.15s** — Wave 28/19/12 band; still ~0.30s above Wave 5 best (~2.84s).
  Not claimed as a new optimization win (correctness/edge waves since 28).
- optimize cold→hot ≈ **29.5×** looks larger than Wave 28 because **cold** was
  slower this run (0.286s vs 0.202s); **hot** ≈0.0097s is in-band with Wave 28
  (~0.0099s). Treat absolute speedups as machine jitter, not edge.
- Wrappee-only same-asof ~**1.55×**; full conformal hit ≈0.0018s; adjacent **false**
  (cache size 2) — same honesty as Wave 12/28.
- Panel ≈ **493×** — in Wave 28 band (~507×).

### Wave 39 engineering notes

- Cross-section edges: empty / length-mismatch / single-name / decile n<10;
  `decile_portfolios` fail-closed on align + `n_buckets`/`min_names`.
- PERF rebench only (above); no vendor/live/parallel-causal work.

## Next levers (not yet shipped)

- Vendor market-data / live broker (explicitly out of overnight scope).
- Parallel causal dates / w_prev (still blocked — sequential required).

## How to re-bench

```bash
MLFLOW_DISABLE_AGENT_HINT=1 .venv/bin/python -c "..."  # clear_panel_cache + clear_forecast_caches / clear_wrappee_cache
uv run pytest -q -m 'not network'
```
