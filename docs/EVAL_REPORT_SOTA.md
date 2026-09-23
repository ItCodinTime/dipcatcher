# SOTA Evaluation Report — return-distribution forecasting

**Status:** contract-v2 results final for the primary cells and the
20-model extended arena; §4.3 daily crossover landed (4h in flight);
seed-11 replication verified under contract v2; seed-23 in flight.
**Scope (declared):** causal return-distribution forecasting on Binance USDT
spot bars (daily + 4-hour). Research-only; `live_pnl_claim: false` in every
receipt. This document is the outsider-auditable artifact; `.dsh-24x7/PROOF.md`
`## SOTA` cites it.

## 1. Hypotheses (declared before looking at results)

- **H1 (primary):** the lab's causal univariate distribution challengers beat
  every published foundation-model forecaster on proper distributional scores
  (CRPS / pinball) on this domain.
- **H2 (robustness):** H1 survives Hansen SPA, Diebold–Mariano, and the
  Hansen–Lunde–Nason Model Confidence Set at α = 0.10 on balanced
  chronological panels.
- **H3 (crossover, in flight):** the ordering survives under the Kronos
  paper's own evaluation protocol (arXiv:2508.02739 App. D): close-path
  RankIC, H-step return RankIC, realized-volatility MAE/R².
- **Non-claim:** no live-trading, execution, capacity, or P&L statement.

## 2. Universe and data

| cell | bars | assets | bar source |
|---|---|---|---|
| daily | last ~999 (v3) / ~4000 deep (v4+) | 11 (v3) / 5 deep (v4) | Binance public klines |
| 4h | ~1500 (v3) / ~4000 deep (h4f) | 5 | same |

Every bar file is sha256-hashed into the receipts (`bars_sha256`); validation
enforces strictly increasing gap-free `event_time`, `available_time ≤ next
bar open`, positive prices, consistent OHLC. Timestamp reconstruction for
legacy shards is hash-verified (`timestamp_source:
reconstructed_from_hash_verified_bars`).

## 3. Protocol (frozen before scoring)

- Walk-forward origins: last `origins_per_asset` bars per cell (150 v3 /
  300 v4+). At origin `i` predictors see `bars[:i+1]` only; target is the
  next bar's close-to-close return.
- Predictive distribution → scores: CRPS (proper, primary) and pinball at
  τ ∈ {0.05, 0.5, 0.95}. Quantile-output targets scored through the
  quantile-pinball identity on a completed quantile function (trapezoid on
  the native grid, exponential tails — disclosed in `quantile_completion`).
- Challenger slate (all causal, trailing-window only):
  `dip_gauss`, `dip_student_t`, `dip_ewma_t`, `dip_empirical`,
  `dip_empirical_long`, `dip_garch_t`, `dip_fhs`, `dip_ewma_emp`,
  `dip_lgbm_q`, `dip_blend`.
- Published targets (zero-shot, as released):
  `kronos_small` (NeoQuasar/Kronos-small + canonical `Kronos-Tokenizer-base`),
  `chronos2` (`autogluon/chronos-2`), `bolt_small`
  (`autogluon/chronos-bolt-small`), `timesfm`
  (`google/timesfm-2.5-200m-pytorch`).
- Statistical battery on the **balanced chronological panel** (per-target-time
  cross-asset mean losses; consecutive timestamps only; ≥20 times required):
  Diebold–Mariano per pair, Hansen SPA per target, MCS @0.10 via stationary
  block bootstrap (n_boot=2000), effect sizes via paired bootstrap intervals,
  block-length sensitivity sweep.

## 4. Results

### 4.1 Primary CRPS eval (contract v2, seed 7)

All TimesFM columns below are contract-v2 (`np.delete(full_forecast, 5)`);
the Kronos columns are canonical `Kronos-Tokenizer-base` pairing. Sources:
`.dsh-24x7/evidence-sota-eval-v3-d1-tfmv2.json`,
`evidence-sota-eval-v4-d1-v2.json`, `evidence-sota-eval-h4f-v2.json`.

**v4 deep-daily** (5 assets × 300 origins, balanced panel n=300):

| model | CRPS |
|---|---|
| dip_fhs | 0.014945 |
| dip_garch_t | 0.014981 |
| dip_blend | 0.015095 |
| dip_empirical | 0.015099 |
| dip_student_t | 0.015103 |
| dip_empirical_long | 0.015150 |
| dip_ewma_emp | 0.015192 |
| dip_ewma_t | 0.015293 |
| dip_gauss | 0.015438 |
| dip_lgbm_q | 0.015451 |
| **timesfm** | **0.015917** |
| **chronos2** | **0.016390** |
| **bolt_small** | **0.017857** |
| **kronos_small** | **0.021228** |

MCS @0.10 = {fhs, garch_t, blend, empirical, student_t} — **all four
targets excluded**; SPA p_upper at floor (≤0.001) per target; every DM
pair favors the challenger (weakest: timesfm vs dip_lgbm_q, p=0.0204).

**v3 daily** (11 assets × 150 origins, balanced n=150): identical ordering —
all 8 challengers (0.013987–0.014504) < timesfm-v2 0.014860 < chronos2
0.016025 < bolt 0.018695 < kronos 0.023339; MCS={6 challengers}, all
targets excluded, SPA p_upper ≤ 0.002.

**h4f 4-hour full-coverage** (5 assets × 300 origins, **1500/1500
complete**, balanced n=295): all 10 challengers (0.005218–0.005475)
< timesfm-v2 0.005503 < bolt 0.005748 < chronos2 0.006035
< kronos 0.007057; MCS={6 challengers}, all targets excluded,
SPA p_upper ≤ 0.007.

**Extended challenger arena (contract v2, inference computed)** —
`merge_d1_v2aug.json`, `merge_h4f_v2aug.json`: six additional challengers
spliced onto verified shards (dip_gmm_k, dip_skt, dip_qar, dip_conf_t,
dip_regime, dip_stack; 20 models total). Daily deep-5 MCS @0.10 =
{fhs, garch_t, regime, qar, conf_t, blend, empirical, student_t}; 4h MCS
includes dip_stack (3rd, 0.005273) plus 13 other challengers. No single
challenger separates from the fhs/garch_t leaders — the frontier is a
statistical tier; every published model is outside it at both horizons.

### 4.2 Coverage

- v4 daily: 1500/1500 complete origins; per-model/per-asset
  `coverage_fraction = 1.0` for every model.
- 4h: `dip_student_t` legacy fit failed on flat windows → complete-case
  shards (n=420/750 v3). Hardened fitter (scaled MLE + MoM fallback) yields
  **0 NaN across all 1500 h4f origins** — the v2 4h inference runs on the
  complete matrix, no case deletion.
- `dip_lgbm_q`: 300/300 on every cell after warmup handling; warmup
  behavior disclosed in §6.

### 4.3 Published-protocol crossover (Kronos App. D)

`scripts/sota_eval_native.py` fleet — per-sample close-path RankIC,
pooled H-step return RankIC, realized-vol MAE/R²; test window 2024-07-01+;
lookback/horizon per paper Table 8 (1d: 40/12; 4h: 90/18). Deviations
disclosed: close channel only (paper averages OHLC); Binance tape, not
the paper's unreleased qlib-format data; challenger paths are honest
distributional constructions (mean path `p_t(1+μ̂)^h`, vol forecast
`(H-1)σ̂²`) — flat mean paths yield undefined path-RankIC for some
challengers (disclosed, not hidden).

**Daily — LANDED** (`merge_native_d1.json`, 11 assets × 300 origins,
corrected 40/12 verified in every receipt):

| model | path RankIC | ret RankIC | vol MAE |
|---|---|---|---|
| **dip_garch_t** | **0.0402** | −0.0113 | 0.0143 |
| timesfm | 0.0296 | **+0.0040** | 0.0165 |
| kronos_small | 0.0127 | +0.0026 | 0.0157 |
| dip_lgbm_q | 0.0062 | −0.0194 | **0.0116** |
| bolt_small | −0.0070 | −0.0341 | 0.0165 |
| chronos2 | −0.0332 | −0.0595 | 0.0165 |

(14 challengers + 4 targets scored; table shows leads per metric.)
Reading, honestly: `dip_garch_t` wins the paper's headline path-shape
metric outright; challengers take vol MAE (best `dip_lgbm_q` 0.0116,
`dip_ewma_emp` 0.0134 vs kronos 0.0157); published models retain the
direction-of-return metric (timesfm +0.0040, kronos +0.0026 vs dip best
+0.0002). `dip_student_t` vol-R² −5.59 is a real failure (disclosed);
`dip_ewma_t` path-RankIC is undefined by construction (flat mean path).

**4h — IN FLIGHT** (`nh_*` shards at 150/300 origins, corrected 90/18
confirmed live; two stale 80/12 receipts on disk will be overwritten).

### 4.4 Robustness (`evidence-sota-robustness.txt`)

- MCS membership invariant across block lengths {1,2,6,12.1,24.2,120.1}
  and bootstrap seeds {7,11,23,42,101}.
- Aligned-panel SPA `p_upper` ≤ 0.003 for every target (floor), seeds swept.
- Effect sizes with simultaneous CIs: regenerated in each v2 merge receipt
  (`effect_sizes`).
- Seed replication (`merge_s11_v2aug.json`, seed 11, 5 deep-daily assets ×
  300 origins, verified contract — s11 timesfm proven bitwise-equal to the
  v2 tfmfix rerun before restamping): deterministic models bit-identical
  to seed 7; `kronos_small` resampled 0.021200 vs 0.021228; the 20-model
  MCS superior set at seed 11 is **identical** to seed 7 ({fhs, garch_t,
  regime, qar, conf_t, blend, empirical, student_t}); all four targets
  excluded again. Seed-23 corrected-pairing fleet (5 assets × 300,
  v2-native) in flight at time of writing.

## 5. Reproduction

```bash
# merge + inference is pure numpy — no models needed
.venv/bin/python scripts/sota_eval_kronos.py \
  --merge-parts .dsh-24x7/eval-full/<cell>*.npz \
  --bars-root data/raw/sources --merge-out /tmp/merged.json
# native-protocol crossover merge
.venv/bin/python scripts/sota_eval_native.py \
  --merge-parts .dsh-24x7/native/nd_*.npz --merge-out /tmp/native.json
```

Every receipt embeds: bar-file hashes, per-model artifact hashes, protocol
config, implementation hashes (`sota_eval_kronos.py`, `inference.py`,
`snooping.py`, `sota_evidence.py`), and `source_parts_sha256`.

## 6. Incident & limitation log (nothing hidden)

1. **Kronos tokenizer pairing** — first v4 fleet paired Kronos-small with
   `Kronos-Tokenizer-2k` (non-canonical) → 2.5–6× degraded; corrected
   `Tokenizer-base` rerun spliced (`splice_kronos_fix.py`), still loses.
2. **TimesFM channel-5 incident (2026-09-22)** — upstream `full_forecast`
   layout `[q50,q10..q40,point,q60..q90]`; legacy remote adapter scored
   `q[:9]` (dropped q90, injected point → ~0.8% CRPS inflation on BTC
   probe); local v1 scored `q[1:]` (dropped q50, injected point — near-
   harmless). Contract v2 = `np.delete(q,5)`. Corrected probe: timesfm
   0.0120 vs challengers 0.0111–0.0117 — ordering survives.
3. **`dip_student_t` 4h coverage** — legacy `scipy.stats.t.fit` failed on
   flat windows (≤63% of origins on some assets); hardened fitter (×100
   scaled MLE + MoM fallback) yields 0 NaN/1500 in the `h4f_*` v2 cell —
   resolved, inference runs on the complete matrix.
4. **Seed-11/23 legacy shards carry the wrong Kronos pairing** — replicate
   only the handicapped run; corrected-pairing fleets landed/running:
   `s11_*` done and verified-v2 (timesfm column bitwise-equal to the
   tfmfix rerun — adapter fix predates the contract field in the spawned
   checkout); `s23_*` (5 deep assets × 300, seed=23) in flight.
5. **Native-protocol freq misclassification (2026-09-22)** — `sota_eval_native`
   converted the bar interval ns→hours with divisor 3.6e15 (off by 1000×;
   correct: 3.6e12), so every file classified `freq="1h"` → all 16 first-pass
   jobs ran lookback 80 / horizon 12 instead of the paper's 40/12 (1d) and
   90/18 (4h). Caught on receipt audit (`lookback: 80` not in the paper
   table). Fixed + respawned with explicit `--lookback/--horizon` flags
   (`spawn_native4.ps1`); first-pass receipts are superseded (overwritten).
5b. **lgbm warmup** — `dip_lgbm_q` needs ≥ ~60 trailing rows; partial coverage
   on short 4h history is disclosed per-asset in `coverage`.
6. **Scope** — Binance USDT spot only; no cross-market claim. Phase G
   decision: **G-bound** (2026-09-22). The claim is permanently scoped to
   *causal return-distribution forecasting on Binance USDT spot bars,
   daily + 4-hour*. Scoped-and-complete beats broad-and-partial under the
   lab's honesty contract; a second-domain fleet would be a new claim, not
   a completion of this one.
7. **Multi-horizon (Phase D) decision** — deferred-with-reason: the shipped
   challengers are one-step distributional forecasters; h-step CRPS would
   require new estimators designed after the freeze (post-hoc model
   invention). Multi-step forecasting *is* evaluated under the published
   protocol instead (§4.3: H-step path RankIC, return RankIC, vol MAE/R²
   at H=12 daily / H=18 4h). If a future need arises for h-step CRPS, it
   requires a declared challenger extension spec before any scoring.

## 7. Verdict

- [x] contract-v2 merged receipts on all cells, challengers retained, all
      targets excluded from MCS @0.10 (v3-d1, v4-d1, h4f all confirmed)
- [~] native-protocol ordering reproduced or deviation documented —
      daily LANDED (dip_garch_t wins path RankIC outright; published
      models keep ret-direction RankIC — split verdict, §4.3); 4h in
      flight (`nh_*` at 150/300)
- [x] multi-horizon cells (Phase D): explicitly bounded — see incident 7;
      multi-step evidence lives in the crossover protocol instead
- [x] scope decision (Phase G): G-bound — Binance USDT crypto
      return-distribution forecasting, daily + 4h
- [ ] status line in PROOF.md updated with final scope
