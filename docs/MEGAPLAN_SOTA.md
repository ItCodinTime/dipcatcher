# MEGAPLAN — Bulletproof SOTA for return-distribution forecasting

**Status:** active. **Owner lane:** forecasting-SOTA megaplan (distinct from the
`MEGAPLAN_SHARPE5.md` strategy-performance lane and the `.dsh-24x7`
Industry-grade incumbent lane).

## What is already proven (baseline, do not regress)

`.dsh-24x7/PROOF.md` `## SOTA` is `STATUS: PROVEN` **scoped**:

- Causal next-bar return-distribution forecasting on Binance USDT spot,
  daily + 4h.
- Two independent fleets (v3: 150 origins/cell; v4: 300 origins/cell, deep
  bars), all four published foundation models excluded from MCS @0.10
  (Kronos-small/base, Chronos-2, Chronos-Bolt-small, TimesFM-2.5); best
  challengers `dip_garch_t`/`dip_fhs`; seed-11/23 replications; adversarial
  audit + local macOS reproduction documented.

This is real, but it is *our* dataset, *our* horizons, *our* protocol. The
remaining objections a skeptic can raise are enumerated below — each maps to a
phase. The megaplan is complete when every objection either has a receipt or is
declared permanently out-of-scope in PROOF.md.

## Objection map → phases

| objection | phase |
|---|---|
| "You picked the dataset and metric" | C (published-protocol crossover) |
| "Next-bar only; real forecasting is multi-step" | D (horizon generalization) |
| "Coverage holes — student_t NaN'd on 4h, lgbm partial" | A (coverage completeness) |
| "SPA p_upper doesn't reject; seeds under-covered" | B (statistical bulletproofing) |
| "One lab, one machine ran it" | E (independent replication) |
| "Crypto only — SOTA claims broader" | G (scope decision: expand or bound) |
| "No paper-grade writeup an outsider can audit" | F (adjudicated report) |

## Phase A — Coverage completeness (small, mechanical)

- **A1** ✅ done 2026-09-22: hardened `_fit_student_t` yields 0 NaN; full-
  coverage `h4f_*` fleet (5×300 deep bars, canonical tokenizer) complete and
  merged (`evidence-sota-eval-h4f-v2.json`: 1500/1500 complete, balanced
  n=295, all targets MCS-excluded — see §4.1 of EVAL_REPORT_SOTA.md).
- **A2** ✅ resolved by the same shard: `dip_lgbm_q` 0/300 NaN on 4h-deep bars
  (the earlier partial coverage was short-history warmup, eliminated by deep
  bars' longer runway — disclosed warmup threshold stands).
- **A3** ✅ done (already in `coverage_summary`): every merged receipt carries a
  per-model, per-asset coverage table (requested/emitted/finite_crps/
  scored_all_metrics/missing_or_failed/coverage_fraction). Verified on
  `evidence-sota-eval-v4-d1-repro.json` 2026-09-22.
- **A5** ⚠️ TimesFM scoring-contract incident (2026-09-22): upstream emits
  channels `[q50, q10..q40, point, q60..q90]` (ch5 = point). Legacy remote
  adapter scored `q[:9]` (drops q90, injects point); local v1 scored `q[1:]`
  (drops q50, injects point). Both inject the point as a pseudo-quantile.
  Fixed to `np.delete(q, 5)` = contract v2; remote script + `sota_eval_native`
  patched; native fleet respawned (`spawn_native3`). **Resolved 2026-09-22:**
  tfmfix fleet (30 shards) complete → `splice_timesfm_fix.py` spliced v2
  columns into `.v2.npz` (deterministic challenger columns verified
  bit-identical) → v3-d1, v4-d1 (with canonical-Kronos `.fixed`), h4f all
  re-merged under contract v2 — all targets MCS-excluded everywhere.
- **A4** ⚠️ provenance audit 2026-09-22: `seed11_*`/`seed23_*` shards carry the
  WRONG Kronos tokenizer pairing (2k, not base — kronos CRPS ~0.046 vs corrected
  ~0.0164) and lack scoring-contract meta → they replicate the *handicapped*
  run only. Corrected-pairing seed replication = `s11_*` fleet queued
  (`scripts/spawn_s11.ps1`, seed 11, canonical pairing, 5 deep-daily assets).

## Phase B — Statistical bulletproofing

- **B1** Seeds: ✅ seed-11 done 2026-09-22 (`evidence-sota-eval-s11-v2.json`):
  corrected-pairing `s11_*` fleet (5 deep-daily × 300, contract-v2 shards
  verified via producing-script sha256) — challengers bit-identical to seed-7
  (deterministic), kronos_small resampled 0.021200 vs 0.021228; MCS={5
  challengers}, all targets excluded, SPA floor. Seed-23 optional (4h cells
  remain seed-7 only — disclosed).
- **B2** Origins: v4 daily is at 300/asset; pushing to the ~500+ the deep bars
  allow shrinks DM/SPA/MCS uncertainty — the only path to moving `p_upper`
  ≈0.5 (it is the bootstrap floor under correlation; more independent origins
  is the lever, not more bootstraps).
- **B3** ✅ done 2026-09-22 (`evidence-sota-robustness.txt`): MCS already uses a
  stationary (Politis–White) block bootstrap; membership invariant across block
  lengths 1→24 and seeds {7,11,23,42,101}; SPA `p_upper` ≤ 0.003 on the aligned
  daily panel (floor-level, better than the pooled-panel ~0.5 disclosure).
- **B4** ✅ done 2026-09-23 (`scripts/sota_effect_sizes.py`, receipts
  `evidence-sota-effect-sizes-{d1,h4}.json`, computed on `codex-remote`):
  per-origin CRPS deltas, per-pair complete-case, stationary-bootstrap 95% CIs
  (Politis–White block). d1: 39/40 pairs CI>0 (only `timesfm−dip_lgbm_q`
  crosses zero: +1.3% mean, weakest challenger vs strong target); h4: 40/40.
  dip_garch_t vs targets: +82%/+84% (kronos), +11%/+14% (chronos2),
  +21%/+9% (bolt), +4%/+7% (timesfm) d1/h4.

## Phase C — Published-protocol crossover (the strongest single upgrade)

Today we beat targets on *our* benchmark. Phase C beats them on *theirs*:

- **C1** 🔄 in flight 2026-09-22: protocol extracted from arXiv:2508.02739
  App.D (test window Jul-2024+, per-freq lookback/horizon table, close-channel
  path RankIC + return RankIC + realized-vol MAE/R²). Harness:
  `scripts/sota_eval_native.py`; 16-asset fleet `nd_*`/`nh_*` on remote.
  ⚠️ freq-inference bug (ns→hours divisor off 1000×) made the first pass run
  lookback 80/horizon 12 on every file — caught via receipt audit
  (`lookback: 80`), fixed, respawned (`spawn_native4.ps1`, explicit flags).
  Deviation disclosed in receipts: close channel only (paper averages OHLC).
  Their per-asset numbers are figure-only → we reproduce the *protocol* and
  compare model orderings + magnitudes, not cell-exact values.
- **C2** Their eval harness is `finetune/qlib_test.py` (qlib-format .bin data,
  not publicly released) → crossover uses our hash-verified Binance bars under
  their protocol; noted as same-protocol-different-tape.
- **C3** Chronos/TimesFM publish no financial eval protocol → crossover is
  Kronos-protocol only (all four targets still scored under it).

This phase is what converts "SOTA on our benchmark" into "SOTA on the
literature's benchmark". It is the highest-value item in the plan.

## Phase D — Horizon generalization

- **D1** Extend the eval from next-bar to h-step distributions: h ∈ {1, 5, 20}
  daily (and {1, 6} on 4h). Score the predictive distribution of the h-bar
  cumulative return. Targets emit multi-step paths natively; challengers need
  an honest h-step construction (vol scaling + horizon-aware quantiles —
  whatever is already in the lab's forecaster API, no new fitting tricks
  post-freeze).
- **D2** Multi-step must reuse the same walk-forward origins — no cherry-picked
  sub-windows.

## Phase E — Independent replication

- **E1** The local macOS reproduction exists for Kronos-small/BTC (40 origins).
  Extend to a documented end-to-end local shard (one full daily asset, all
  targets, no remote host) — fixes the lgbm/libomp deadlock or keeps the
  stubbed-column disclosure.
- **E2** CI feasibility — checked 2026-09-22: `.github/workflows/ci.yml` runs
  lint/audit/test/smoke on every push. The merge+inference step is pure
  numpy/scipy (~30s) and *can* run in CI — but it needs the `.dsh-24x7/` loss
  matrices + bar parquets committed to the repo (currently untracked). That is
  a repo-policy decision for the user, not an agent call.
- **E3** Optional: a second remote host or a colleague machine run — the
  v3/v4/local triple already covers OS×arch diversity; a second independent
  *operator* would be the gold standard but is not strictly required.

## Phase F — Adjudicated report

- **F1** `docs/EVAL_REPORT_SOTA.md`: pre-registered-style protocol doc —
  hypotheses, universe, origin scheme, scores, multiplicity battery, complete
  tables, all receipts linked, limitations enumerated. This is the artifact an
  outsider reads; PROOF.md cites it.
- **F2** Every number in the report traceable to a receipt + command (the
  existing convention); no number written by hand.
- **F3** Failure-mode appendix: student-t 4h coverage, lgbm warmup, Kronos
  tokenizer-pairing incident + corrected rerun — disclose everything we
  already disclose internally.

## Phase G — Scope decision (explicit, one-line outcome)

Choose exactly one and write it into PROOF.md:

- **G-expand**: add a second domain (e.g., equity dailies via yfinance, or
  hourly crypto) with the same protocol → widened claim, ~1 fleet of work.
- **G-bound**: declare the claim permanently scoped to "Binance USDT crypto
  return-distribution forecasting, daily+4h" — a *complete* claim within scope,
  not a partial one.

Recommendation: **G-bound** unless the product claims non-crypto markets.
Scoped-and-complete beats broad-and-partial under this lab's honesty contract.

## Ordering

```
A (coverage) → B (stats) → C (crossover) → D (horizons) → F (report)
E (replication) can run parallel to C/D — it only needs committed matrices.
G is a decision, not work — make it before Phase F writes the report scope.
```

Phase C is the single highest-impact item: it is the difference between
"we beat SOTA on our benchmark" and "we beat SOTA on theirs".
