# AUDIT P6.4B — model layer: forecast/features/fusion/labels

Scope: `pipeline/forecast/` (history, conformal, covariance, garch, realized,
wrappee, artifacts, decide, state, facade), `features/` (engine,
cross_sectional, indicators, liquidity, cycles, bars, _kernels, metadata),
`fusion/engine.py`, `labels/engine.py`, `labels/forward.py`,
`pipeline/dataset.py`, `pipeline/train/splits.py` — read end-to-end against
the PIT/honesty contract. Complements AUDIT_P64_DIST.md (models/ family),
AUDIT_P62_STATS.md (validation layer), AUDIT_P65_MICRO.md (execution).

ULTRAPLAN item: `P6.4 Model layer: every file vs its cited paper`
(docs/ULTRAPLAN_FRONTIER.md).

Verdicts: `clean` (checked, matches contract), `fixed` (real bug, patch + KAT),
`waived` (intentionally looser semantics — documented).

| file | claim checked | verdict | fix |
|---|---|---|---|
| pipeline/dataset.py | gold panel join carries every label column needed by the train lanes' observed-end purge (`_aligned_label_end_times`) | **fixed**: `label_end_time_*` columns were persisted by the label engine but dropped at the `labs`→panel join — the purge path silently degraded to index arithmetic on sparse panels. Now joined (PR #212) | PR #212 |
| pipeline/forecast/history.py | `history_for_calibration` must return only rows whose h-step labels are realized strictly before `asof` | **fixed**: when fewer than `horizon+1` sessions preceded `asof`, the code fell back to `last = idx - 1`, returning rows whose labels end *after* the decision bar — unrealized labels leaking into `conformal_sets_asof` calibration. Now returns empty; the caller already declines on empty history (fail-closed) (PR #213) | PR #213 |
| pipeline/forecast/artifacts.py | artifact cache identity must be content bytes, not mtime (docstring contract: "mtime is not identity") | **fixed**: `_load_rl_cached` keyed `_RL_POLICY_CACHE` on `st_mtime` while `_load_ranker_cached` / `_load_garch_spec_cached` / `_load_realized_garch_spec_cached` key on SHA-256 of the file — a same-size/`os.utime` rewrite of `rl_*.joblib` could serve a stale policy. Now content-digest keyed (this PR) | this PR |
| pipeline/forecast/conformal.py | decision-bar y never enters calibration; Mondrian tercile sets conditional on vol bucket; cache keyed on frame content + label + asof | clean: train/cal split never half-splits a date; `history_for_calibration` cutoff correct post-fix; `row_hash` content fingerprint prevents id-reuse collisions; `live_pnl_claim=False` stamped | — |
| pipeline/forecast/covariance.py | named estimators (LW/OAS/EWMA/DCC family) must fail closed on family/object/spec mismatch; trailing window uses contiguous complete-case rows | clean: every named path re-verifies `family`/`covariance_object`/`spec` from the fitter's own params — silent substitution impossible; unmeasured paths return a *labeled* diagonal proxy (`unmeasured_reason`), never disguised as the named estimator | — |
| pipeline/forecast/garch.py | asof GARCH refits an *unfitted spec clone* on strictly-prior returns; artifact digest + full-history digest key the cache | clean: persisted fitted params never reused; `_stamp_strictly_before` double-filter; scope gate (`date_level_ret_1` required) fails closed; PIT-universe restriction; missing artifact → None (documented leniency), invalid/empty universe → `PointInTimeError` | — |
| pipeline/forecast/realized.py | same contract for Parkinson Realized GARCH; HF-RV claims rejected | clean: `realized_measure` + `series_scope` + `intraday_realized_variance` all gated; overlay precedence RGARCH → GARCH; missing RGARCH history warns and falls through to GARCH rather than fabricating ranges | — |
| pipeline/forecast/wrappee.py | wrappee family re-selection always on current cal; train MLE reuse only on identical train fingerprint | clean: fingerprint includes content digest of y_train/scale_train — same-mean collisions impossible; fail-closed input guards on alpha/min_coverage/lengths | — |
| pipeline/forecast/decide.py | `forecast_asof`/`optimize_asof` causal end-to-end; no latest-date fallback; stale model → raise not degrade | clean: empty day raises; ranker/RL/calibrator feature contracts verified per call; `w_prev` shape mismatch raises (no silent flat reset); interval-cap bisection respects hard turnover limit, infeasible → `OptimizationInfeasible`; SYNTHETIC stamp when `data.source=="synthetic"` | — |
| fusion/engine.py | `fuse_signals` risk floor `1e-8` | verified deliberate: `test_fuse_signals_risk_floor_fail_closed` pins `risk=0 → out=1e8`; flat-asset `vol_20=0` is legitimate and downstream optimizer caps bound it. NOT a bug — audited and left unchanged | — |
| labels/engine.py + forward.py | labels causal (shift -h), `label_end_time_*` stamped per row | clean: with PR #212 the end-time columns now reach the train lanes | — |
| pipeline/train/splits.py | walk-forward/purge boundaries | clean (cross-checks AUDIT_P62_STATS.md validation layer) | — |
| features/engine.py + cross_sectional.py | feature grid causal; cs_* cross-sectional transforms per-date | clean: ranks/zscores computed within `event_time` group only | — |
| features/indicators.py | TA library causal, NaN warmup, guarded division | clean: all windows end at bar t (diff/EMA/rolling); warmup NaN never fabricated; flat-series conventions documented (RSI 50, %K 50) | — |
| features/liquidity.py | LOT/FHT/PS-gamma/Hasbrouck/GH/Holden tick estimators vs cited papers | clean: regressions and zero-frequency mappings match cited forms; fail-closed on degenerate inputs (zero abs-return, <10 nonzero, non-positive breadth) | — |
| features/cycles.py + _kernels.py | Ehlers FIR quadrature must be causal ("bar t does not see t+1"); Goertzel in published order | clean: every quadrature tap is a lag; numba kernels replicate the Python loop semantics bit-for-bit; trailing partial bar dropped | — |
| features/bars.py | AFML tick/volume/dollar/imbalance/run bars; tick-rule signs carry forward | clean: zero-move signs carry the last nonzero sign; incomplete final bar excluded | — |

## Summary

- Real bugs fixed: 3 —
  1. `panel()` dropped `label_end_time_*` (sparse-panel purge path unreachable) — PR #212.
  2. `history_for_calibration` returned unrealized labels for early `asof` — PR #213.
  3. `_load_rl_cached` cache keyed on mtime vs content digest — this PR.
- Near-miss investigated and rejected: `fuse_signals` 1e-8 risk floor is a
  pinned design contract, not a bug.
- Waived: `history_for_calibration` excludes labels ending *at* `asof`'s own
  session (conservative — decision assumed intra-session; documented).
- The forecast package's own discipline is strong: every artifact load is
  content-addressed, every estimator verifies its own family/object/spec,
  and every "no data" path either raises or returns a labeled proxy.
