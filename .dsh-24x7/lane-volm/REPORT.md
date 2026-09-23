# lane-volm: `dip_volm` — volume-conditioned volatility challenger

**Date:** 2026-09-23 · **Model:** `dip_volm` · **Tool:** `scripts/_volm_col.py` (new)
**Scope:** CRPS + pinball at TAUS=(0.05, 0.5, 0.95), 1-step close-to-close returns,
deterministic origin grid on the v2aug shard family.

## Design (causal, deterministic, fixed hyperparameters)

Per origin i, over the trailing `garch_window=750` returns `r_t` and aligned
volumes `v_t`:

- **Volume surprise** `vs_t = v_t / median(v_{t-19..t})` — 20-bar trailing
  median baseline inclusive of t. Bars with non-finite/non-positive volume (or
  <5 valid baseline bars) emit NaN `vs_t` and are skipped wherever `vs` is used.
- **Recent surprise** `S` = EWMA (lam=0.7, `lam^age` normalized weights) of the
  last 10 window positions' `vs`; invalid positions dropped.
- **Base sigma** = `ewma_next_sigma(r, lam=0.94)` (RiskMetrics); the same
  recursion supplies per-bar forecasts `sig_t` for residual standardization.
- **Elasticity `gamma`** — per-origin OLS slope of `log|r|` on `log(vs)` clipped
  to [0,1] (`gamma_reg`), **blended 50/50 with fixed prior 0.5**:
  `gamma_hat = 0.5·gamma_reg + 0.25`. Falls back to 0.5 when the regression is
  infeasible (<30 usable pairs or degenerate `log vs` variance). The blend is
  fixed and disclosed — not tuned.
- **Conditioned sigma** `sigma_cond = sigma_base · clip(S^gamma_hat, 0.6, 1.8)`.
- **Quantiles** `q_tau = mu_hat + sigma_cond · quantile(z, tau)` at LGBM_TAUS,
  `z_t = r_t / (sig_t · vs_t^gamma_hat)`, `mu_hat = sum(r)/(n+50)` (50-obs
  pseudo-prior shrinkage to 0). Requires ≥30 finite z.
- CRPS via `crps_from_quantiles(y, LGBM_TAUS, q)`; pinball via `qf_at`.
- **Honest-NaN policy:** <50% valid `vs` coverage in window, no valid `vs` in
  the S tail, or <30 finite z → NaN row (disclosed, never imputed).

## Provenance / verification

- All 10 target shards (`d1_*_1d_deep.v2aug.npz`, `h4f_*_4h_deep.v2aug.npz`)
  hash-verified identical local vs remote (`me@100.116.120.51:D:\dipcatcher`)
  before compute; columns computed **remote** (Windows, matching the shard
  fleet's platform), spliced + merged locally.
- Cross-check: `d1_btcusdt` column recomputed locally is bit-identical in
  pinball/timestamps; CRPS differs only at ≤3.5e-18 (BLAS summation order).
- Each column artifact binds `bars_sha256` + protocol fields; splice verified
  bars provenance, row count, protocol, and asset identity per shard.

## Pooled results

### Daily (`merge_d1.json`, 1500 rows, 1500 complete)

| Rank | Model | Pooled CRPS |
|---:|---|---:|
| 1 | dip_fhs | 0.014945 |
| 2 | dip_garch_t | 0.014981 |
| 3 | dip_regime | 0.015065 |
| 4 | dip_qar | 0.015090 |
| 5 | dip_conf_t | 0.015092 |
| 6 | dip_blend | 0.015095 |
| 7 | dip_empirical | 0.015099 |
| 8 | dip_student_t | 0.015103 |
| 9 | dip_stack | 0.015119 |
| 10 | dip_skt | 0.015146 |
| 11 | dip_empirical_long | 0.015150 |
| **12** | **dip_volm** | **0.015170** |
| 13 | dip_gmm_k | 0.015181 |
| 14 | dip_ewma_emp | 0.015192 |
| 15 | dip_ewma_t | 0.015293 |
| 16 | dip_gauss | 0.015438 |
| 17 | dip_lgbm_q | 0.015451 |
| 18 | timesfm | 0.015917 |
| 19 | chronos2 | 0.016390 |
| 20 | bolt_small | 0.017857 |
| 21 | kronos_small | 0.064231 |

- **MCS @0.10: IN** (p = 0.1828). Set: dip_student_t, dip_empirical,
  dip_garch_t, dip_fhs, dip_ewma_emp, dip_blend, dip_skt, dip_qar, dip_conf_t,
  dip_regime, dip_volm.
- Pinball pooled: τ0.05 = 0.003106, τ0.5 = 0.010084, τ0.95 = 0.003630.
- Δ vs **dip_ewma_emp**: −0.0000223 (volm better; wins 4/5 assets — all but
  XRPUSDT). Δ vs **dip_garch_t**: +0.0001887 (worse; loses all 5 assets).

### 4h (`merge_h4f.json`, 1500 rows, 1500 complete)

| Rank | Model | Pooled CRPS |
|---:|---|---:|
| 1 | dip_fhs | 0.005218 |
| 2 | dip_garch_t | 0.005220 |
| 3 | dip_stack | 0.005273 |
| 4 | dip_regime | 0.005284 |
| **5** | **dip_volm** | **0.005285** |
| 6 | dip_ewma_emp | 0.005306 |
| 7 | dip_blend | 0.005312 |
| 8 | dip_gmm_k | 0.005324 |
| 9 | dip_empirical_long | 0.005324 |
| 10 | dip_qar | 0.005325 |
| 11 | dip_empirical | 0.005325 |
| 12 | dip_skt | 0.005327 |
| 13 | dip_conf_t | 0.005327 |
| 14 | dip_student_t | 0.005329 |
| 15 | dip_ewma_t | 0.005375 |
| 16 | dip_gauss | 0.005390 |
| 17 | dip_lgbm_q | 0.005475 |
| 18 | timesfm | 0.005503 |
| 19 | bolt_small | 0.005748 |
| 20 | chronos2 | 0.006035 |
| 21 | kronos_small | 0.007057 |

- **MCS @0.10: OUT** (p = 0.0629, borderline). Note dip_ewma_emp stays in at
  p = 0.1489 despite a worse pooled mean — MCS is driven by per-row loss
  differentials, not pooled means.
- Pinball pooled: τ0.05 = 0.001055, τ0.5 = 0.003447, τ0.95 = 0.001458.
- Δ vs **dip_ewma_emp**: −0.0000218 (volm better; wins 2/5 assets — BNBUSDT,
  XRPUSDT — but by a wide XRP margin). Δ vs **dip_garch_t**: +0.0000644
  (worse; loses all 5 assets).

## Diagnostics (from column meta)

| Shard | NaN CRPS | NaN pin | γ_reg mean | γ_hat mean | vol cover min |
|---|---:|---:|---:|---:|---:|
| d1_bnbusdt | 0 | 0 | 0.716 | 0.608 | 1.000 |
| d1_btcusdt | 0 | 0 | 1.000 | 0.750 | 1.000 |
| d1_ethusdt | 0 | 0 | 1.000 | 0.750 | 1.000 |
| d1_solusdt | 0 | 0 | 0.964 | 0.732 | 1.000 |
| d1_xrpusdt | 0 | 0 | 0.936 | 0.718 | 1.000 |
| h4f_bnbusdt | 0 | 0 | 0.833 | 0.667 | 1.000 |
| h4f_btcusdt | 0 | 0 | 0.957 | 0.729 | 1.000 |
| h4f_ethusdt | 0 | 0 | 0.914 | 0.707 | 1.000 |
| h4f_solusdt | 0 | 0 | 0.957 | 0.729 | 1.000 |
| h4f_xrpusdt | 0 | 0 | 0.871 | 0.686 | 1.000 |

`gamma_reg` saturates the [0,1] clip at 1.0 on BTC/ETH daily — the data-driven
elasticity is high; the 50/50 blend keeps `gamma_hat` in [0.61, 0.75]. Volume
coverage was 100% on every window of every shard: zero honest-NaN rows.

## Verdict (honest)

`dip_volm` is a **mid-pack challenger with a real but small edge over its
unconditioned parent**:

- Beats `dip_ewma_emp` (same EWMA sigma backbone, no volume conditioning) on
  **both** horizons' pooled CRPS — the volume conditioning adds value over the
  backbone it extends.
- Does **not** reach the GARCH-t / FHS frontier (worse than `dip_garch_t` on
  10/10 assets; ~+1.2–1.3% pooled CRPS).
- Best showing at 4h: rank 5/21, ahead of dip_ewma_emp, dip_blend, dip_gmm_k,
  dip_qar — but MCS-excluded at @0.10 (p=0.063, borderline). On daily it is
  MCS-included (p=0.183) yet only 12th by pooled CRPS.
- Zero NaN rows; fully deterministic; ~1 s per shard.

Net: volume surprise carries genuine incremental signal over plain EWMA, but
the magnitude is small relative to what proper conditional-volatility +
filtered-residual models already capture. Candidate ensemble member; not a
standalone SOTA challenger.

## Artifacts

- `scripts/_volm_col.py` — column generator (new file; no existing files edited).
- `.dsh-24x7/lane-volm/cols/*.volm.npz` — 10 column artifacts (meta records
  `volm_params`, `volm_stats`, shard + bars sha256).
- `.dsh-24x7/lane-volm/spliced/*.v2aug.volm.npz` — 10 spliced shards (21 cols).
- `.dsh-24x7/lane-volm/merge_d1.json` + `merge_d1.losses.npz` — pooled daily receipt.
- `.dsh-24x7/lane-volm/merge_h4f.json` + `merge_h4f.losses.npz` — pooled 4h receipt.
- Compute: remote `me@100.116.120.51:D:\dipcatcher` via `run_volm.ps1`
  (shard hashes verified vs local before compute).
