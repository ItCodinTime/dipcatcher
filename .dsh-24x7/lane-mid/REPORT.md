# lane-mid: `dip_mid` challenger report

Date: 2026-09-23. Arena: verified SOTA shards `d1_*_1d_deep.v2aug.npz` +
`h4f_*_4h_deep.v2aug.npz` (5 assets each, 300 origins/asset, TAUS=(0.05,0.5,0.95),
scoring contract `native_shapes_timesfm_point_first.v2+spliced`).

## Model

`dip_mid` — GARCH-MIDAS-style mixed-frequency challenger
(`scripts/_mid_col.py`, mirrors `_challenger_col.py`/`_egarch_col.py` conventions):

- fast vol: `sigma_short` = RiskMetrics EWMA (lam=0.94) next-bar forecast over
  the trailing 10 returns of the `garch_window` (750) window; per-bar path
  `sigma_short_t` gives FHS standardized residuals `z_t = r_t / sigma_short_t`.
- long vol (MIDAS filter): last `K*bpc` returns grouped into consecutive
  `bpc`-bar blocks (`bpc=6` for 4h bars → daily; `bpc=5` for 1d bars → weekly;
  consecutive-index bucketing, valid because `validate_bars` guarantees
  gap-free bars). `c_j` = per-bar RMS of block j.
- `sigma_cond = sigma_short * clip(mean(c[-30:]) / mean(c_all), 0.7, 1.5)` —
  regime tilt, not level replacement.
- shape: `mu = 0.5*mean(rets_long)`; `q(levels) = mu + sigma_cond *
  quantile(z, levels)` at LGBM_TAUS (9 pts); CRPS via `crps_from_quantiles`;
  TAUS read off via `qf_at`. Honest NaN if <30 finite z or degenerate vol.

## Provenance / verification

- All 10 remote shards (D:\dipcatcher, me@100.116.120.51) sha256-verified
  byte-identical to local before compute; all 10 matched.
- Code parity remote vs local: `sota_eval_kronos.py`, `_challenger_col.py`,
  `sota_evidence.py`, and both bars files byte-identical. `scoring.py`
  differed only in the placement of the `GARCH_ONE_STEP_CRPS_TAUS` constant
  (functions identical — verified by diff).
- Columns computed on the remote Windows venv, then spliced locally.
  Cross-check: remote `h4f_btcusdt` column vs local smoke run differs only at
  machine epsilon (max |dCRPS| = 1.7e-18, rel ~2.5e-16; `target_time_ns`
  identical) — np.convolve/quantile BLAS rounding, not a logic divergence.
- Splice required `dip_mid` in `KNOWN_MODELS`; applied as a runtime patch to
  `splice_challenger_column.py` (no file edits — repo untouched except the new
  `scripts/_mid_col.py` and this lane's artifacts). All splice checks
  (bars_sha256, protocol fields, asset identity, row count) passed.

## NaN / coverage

300/300 finite CRPS on all 10 shards (0 NaN rows). Tilt clip hits:
d1 — bnb 0 lo/0 hi, btc 0/0, eth 0/0, sol 31/0, xrp 72/0;
h4f — bnb 33/0, btc 22/0, eth 0/0, sol 80/0, xrp 0/13.
Mean tilt < 1 on 9/10 shards (0.83–0.95): recent-30-block coarse vol sits
below the full-window coarse mean over most of these grids, so the tilt mostly
*shrinks* the EWMA sigma. Only h4f_xrp tilted up (mean 1.099, 13 hi-clips).

## Pooled results (1500 rows/lane)

### d1 lane (21 models)

| rank | model | pooled CRPS | MCS @0.10 |
|---|---|---|---|
| 1 | dip_fhs | 0.014945 | in |
| 2 | dip_garch_t | 0.014981 | in |
| 3 | dip_regime | 0.015065 | in |
| … | (dip_qar, dip_conf_t, dip_blend, dip_empirical, dip_student_t, dip_stack, dip_skt, dip_empirical_long, dip_gmm_k, dip_ewma_emp, dip_ewma_t, dip_gauss, dip_lgbm_q) | 0.015090–0.015451 | mixed |
| **17** | **dip_mid** | **0.015602** | **out (p=0.0025)** |
| 18 | timesfm | 0.015917 | out |
| 19 | chronos2 | 0.016390 | out |
| 20 | bolt_small | 0.017857 | out |
| 21 | kronos_small | 0.064231 | out |

### h4f lane (21 models)

| rank | model | pooled CRPS | MCS @0.10 |
|---|---|---|---|
| 1 | dip_fhs | 0.005218 | in |
| 2 | dip_garch_t | 0.005220 | in |
| 3 | dip_stack | 0.005273 | in |
| … | (dip_regime, dip_ewma_emp, dip_blend, dip_gmm_k, dip_empirical_long, dip_qar, dip_empirical, dip_skt, dip_conf_t, dip_student_t, dip_ewma_t, dip_gauss, dip_lgbm_q) | 0.005284–0.005475 | mixed |
| **17** | **dip_mid** | **0.005491** | **out (p=0.0005)** |
| 18 | timesfm | 0.005503 | out |
| 19 | bolt_small | 0.005748 | out |
| 20 | chronos2 | 0.006035 | out |
| 21 | kronos_small | 0.007057 | out |

## Delta vs references (paired DM on the merged loss matrix, n=1500)

| lane | vs | mean dCRPS (mid − ref) | t | p | preferred |
|---|---|---|---|---|---|
| d1 | dip_fhs | +0.000657 (+4.40%) | +7.42 | 0.0000 | dip_fhs |
| d1 | dip_garch_t | +0.000622 (+4.15%) | +7.03 | 0.0000 | dip_garch_t |
| d1 | dip_ewma_emp | +0.000410 | +4.38 | 0.0000 | dip_ewma_emp |
| d1 | dip_stack | +0.000483 | +4.44 | 0.0000 | dip_stack |
| h4f | dip_fhs | +0.000273 (+5.23%) | +5.89 | 0.0000 | dip_fhs |
| h4f | dip_garch_t | +0.000270 (+5.17%) | +6.20 | 0.0000 | dip_garch_t |
| h4f | dip_ewma_emp | +0.000184 | +3.78 | 0.0002 | dip_ewma_emp |
| h4f | dip_stack | +0.000218 | +3.93 | 0.0001 | dip_stack |

dip_mid does beat all four deep targets in both lanes (e.g. h4f vs timesfm
+0.000012, p=0.86 — statistically tied; d1 vs timesfm +0.000314, p=0.08), and
beats bolt/chronos2/kronos clearly, but it never threatens the challenger pack.

## Per-asset mean CRPS (dip_mid)

d1: BNB 0.012600, BTC 0.012639, ETH 0.017092, SOL 0.018385, XRP 0.017296.
h4f: BNB 0.004320, BTC 0.003763, ETH 0.005164, SOL 0.006201, XRP 0.008006.
Pinball (d1): q05 0.003411, q50 0.010047, q95 0.004318.
Pinball (h4f): q05 0.001253, q50 0.003443, q95 0.001659.

## Verdict

**Rejected.** dip_mid ranks 17/21 in both lanes, is excluded from both
MCS@0.10 sets, and is significantly worse than every vol-conditioned
challenger it was designed to extend (dip_fhs/dip_garch_t/dip_ewma_emp/
dip_stack, all p<0.001). The coarse-scale MIDAS tilt is mostly <1 on these
grids (recent coarse vol below its own long-run mean), so it systematically
narrows the FHS-based forecast — and narrower intervals lose CRPS here. The
mixed-frequency idea is not rescued by the clip bounds; a plain
EWMA+empirical-quantile (dip_ewma_emp) dominates it at lower complexity.

## Artifacts

- `scripts/_mid_col.py` — column generator (new file; no existing files edited)
- `.dsh-24x7/lane-mid/cols/{shard}.midcol.npz` — 10 column artifacts
  (crps_col, pin_cols, target_time_ns, meta with tilt/clip stats)
- `.dsh-24x7/lane-mid/spliced/{shard}.v2aug.npz` — 10 spliced shards (21 cols)
- `.dsh-24x7/lane-mid/merge_d1.json` / `merge_d1.losses.npz`
- `.dsh-24x7/lane-mid/merge_h4f.json` / `merge_h4f.losses.npz`
- remote runner: `D:\dipcatcher\.dsh-24x7\lane-mid\_mid_run.ps1`
