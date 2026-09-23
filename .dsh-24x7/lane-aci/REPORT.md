# Lane `aci` — `dip_aci` (Adaptive Conformal Inference over `dip_garch_t`)

**Date:** 2026-09-23 · **Script:** `scripts/_aci_col.py` (new, standalone) · **Method:** Gibbs & Candès (2021, NeurIPS) ACI
**Base:** `dip_garch_t` (GARCH(1,1)-t, `_arch_fit` on `garch_window` returns) · **γ=0.02, clip=[0.01,0.99], warmup=30 origins**

## Design

Per origin `i` on the shard's deterministic grid (300 origins/asset, causal):

1. Compute the base GARCH-t quantile vector at `LGBM_TAUS` (9 levels) — identical
   code path to `_stack_base_quantiles("dip_garch_t", rets_long)`.
2. Maintain an adjusted level `alpha_g` per grid level, init `alpha_g = g`.
3. Emit `q_adj[g] = qf_at(LGBM_TAUS, q_base, alpha_g)` (linear interp inside the
   grid, disclosed exponential tails outside). First 30 origins emit raw `q_base`
   (alphas still adapt).
4. Score: `crps_from_quantiles(y, LGBM_TAUS, q_emit)`; pinball at TAUS levels.
5. **After** scoring: `alpha_g ← clip(alpha_g + γ·(g − 1{y ≤ q_adj[g]}), 0.01, 0.99)`.
   Base-fit failure → honest NaN row, alphas untouched. 0 NaN rows on all shards.

Column artifacts also carry `base_crps_col`/`base_pin_cols` (raw base scored under
the **same** quantile-grid convention) and `alpha_path` for honest deltas.

## Provenance

- Columns computed **remote** (`me@100.116.120.51`, `D:\dipcatcher`, `.venv\Scripts\python.exe`,
  runner `scripts/_run_aci.ps1`). All 10 target shard sha256 verified identical local↔remote
  before compute. `arch 8.0.0` both sides.
- Spliced with `scripts/splice_challenger_column.py` (dip_aci pre-registered in `KNOWN_MODELS`).
- Merged with `scripts/sota_eval_kronos.py --merge-parts --seed 7 --n-boot 2000 --bars-root
  data/raw/sources` (timestamps hash-recovered; `timestamp_source=reconstructed_from_hash_verified_bars`).
- Artifacts: `.dsh-24x7/lane-aci/{<shard>.acicol.npz}`, `spliced/{<shard>.aci.npz}` (21 cols),
  `merge_d1.json`, `merge_h4f.json` (+ `.losses.npz` twins).

## Headline numbers

| Arena | dip_aci CRPS | Arena rank | dip_garch_t CRPS | Δ vs garch_t (as stored) | Δ vs base, same convention, post-warmup | MCS @0.10 |
|---|---|---|---|---|---|---|
| d1 (1500 rows) | **0.015080** | 4th / 21 (of 17 challengers: 4th) | 0.014981 | +0.66% worse | **+0.50% worse** (DM t=2.25, **p=0.025**) | **IN** (p=0.277) |
| h4f (1500 rows) | **0.005247** | 3rd / 21 (of 17: 3rd) | 0.005220 | +0.52% worse | **+0.37% worse** (DM t=1.04, p=0.30) | **IN** (p=0.628) |

"dip_garch_t as stored" uses closed-form `crps_student_t`; `dip_aci` is scored via the
quantile-grid trapezoid (`crps_from_quantiles`) like `dip_qar`/`dip_stack`. The honest
like-for-like column is the same-convention delta: ACI **hurt** CRPS on 9 of 10 shards
(only h4f_btcusdt improved, −0.20%).

## Per-asset, post-warmup CRPS (rows ≥30, same-convention base)

| shard | dip_aci | base grid | Δ |
|---|---|---|---|
| d1_bnbusdt | 0.011865 | 0.011824 | +0.35% |
| d1_btcusdt | 0.012289 | 0.012172 | +0.96% |
| d1_ethusdt | 0.016692 | 0.016632 | +0.36% |
| d1_solusdt | 0.017958 | 0.017898 | +0.34% |
| d1_xrpusdt | 0.016752 | 0.016652 | +0.60% |
| h4f_bnbusdt | 0.004443 | 0.004402 | +0.93% |
| h4f_btcusdt | 0.003723 | 0.003730 | **−0.20%** |
| h4f_ethusdt | 0.005068 | 0.005065 | +0.05% |
| h4f_solusdt | 0.006301 | 0.006251 | +0.80% |
| h4f_xrpusdt | 0.007767 | 0.007751 | +0.20% |

## Pinball (pooled, post-warmup Δ vs same-convention base)

- **τ=0.05: ACI better on 8/10 shards** (d1 pooled 0.003010 vs garch_t-stored 0.003139;
  up to −0.00023 per-asset). The left tail was genuinely undercovered and ACI fixed it.
- **τ=0.50: worse on all 10** (median resampling adds noise, no systematic fix).
- **τ=0.95: worse on 8/10** — upper-tail alpha drifts to the 0.99 clip on several
  shards and the exponential-tail extrapolation overshoots the pinball optimum.

## Alpha drift diagnostic (mean |alpha_t − tau| per level, all origins)

| level | d1 mean (5 shards) | h4f mean (5 shards) | typical final direction |
|---|---|---|---|
| 0.05 | 0.031 | 0.017 | ↑ drifts to 0.07–0.16 (base left tail too thin) |
| 0.50 | 0.034 | 0.038 | wanders ±0.06 around 0.5 |
| 0.95 | 0.019 | 0.016 | ↑ to 0.95–0.99 clip on 6/10 (base right tail thin) |

`alpha_path` shows no convergence — as designed for drift tracking, α keeps
random-walking with step γ=0.02 per coverage event (e.g. ETH-d1 α0.05:
0.06→0.16 over the run). The persistent drift direction is a real signal that
GARCH-t tails are thin on crypto returns, but the constant-step recursion
injects enough level noise to cost more CRPS than calibration recovers.

## Verdict

**Negative, honestly reported.** `dip_aci` is a legitimate published method,
implemented causally (update strictly after scoring) and verified end-to-end —
but on this grid it **does not improve on raw `dip_garch_t`**: +0.50%/+0.37%
post-warmup CRPS (d1 significant at p=0.025; h4f n.s.). It lands in the MCS on
both arenas (it is still a garch_t-family model) at rank 4 (d1) / 3 (h4f), but
that rank is inherited from the base. The asymmetric result — pinball@0.05
improved while CRPS/0.5/0.95 degraded — says the base's real defect is
tail-thinness at the extremes; a full-quantile-path correction (or smaller γ /
EWA-decayed gamma, or applying ACI only at the tails) would be needed to
monetize it. `dip_fhs` remains the CRPS leader in both arenas.

## Files

- `scripts/_aci_col.py` — standalone column tool (copied dip_garch_t branch,
  imports `_arch_fit`/`qf_at`/`crps_from_quantiles`/`LGBM_TAUS`/`TAUS`/`_sha256`
  from `sota_eval_kronos`, `pinball_loss` from scoring, `validate_bars`).
- `scripts/_run_aci.ps1` — remote runner (also on `D:\dipcatcher\scripts\`).
- `.dsh-24x7/lane-aci/*.acicol.npz` — 10 column artifacts (incl. `alpha_path`,
  `base_crps_col`, `base_pin_cols`, ACI config + drift diagnostics in meta).
- `.dsh-24x7/lane-aci/spliced/*.aci.npz` — 10 spliced shards (21 model cols).
- `.dsh-24x7/lane-aci/merge_{d1,h4f}.{json,losses.npz}` — merge receipts.
