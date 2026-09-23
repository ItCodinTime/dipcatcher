# dip_xbeta — cross-asset-conditioned challenger

**Model.** First cross-asset challenger in the arena. For target asset A at
origin `i`: trailing simple returns `r_A` over `garch_window` (750); covariate
asset = BTCUSDT for all targets except BTCUSDT itself, which uses ETHUSDT
(same-interval `*_deep.parquet`, inner-joined on `event_time`). Rolling OLS
beta of `r_A` on `r_M` over the trailing 250 aligned obs (min 60). Forecast
`sigma = sqrt(max(0.15, 0.6)·(β·σ_M)² + 0.4·σ_idio²)` with `σ_M`/`σ_idio` the
EWMA(λ=0.94) next-bar sigmas of covariate returns and of residuals
`r_A − α − β·r_M`; mean = 0.5·(α + β·E[r_M]) (50% shrink toward zero).
Quantiles on `LGBM_TAUS`: Student-t with df fitted on residuals (clipped
[3,30], unit-variance scaling `sqrt((ν−2)/ν)`); empirical-quantile fallback on
standardized residuals (never needed — 3000/3000 origins used the t-path).
CRPS via `crps_from_quantiles` on the `LGBM_TAUS` grid (same path as
`dip_qar`); pinball at TAUS via `qf_at`. Fully deterministic, no RNG.

**Honest-failure rule.** Rows where <90% of the window's target timestamps
have a covariate match are left NaN. Observed minimum overlap was 0.995
(BNBUSDT/XRPUSDT 4h); 0 NaN rows in 3000. Covariate file + sha256 recorded per
column as `aux_bars` / `aux_bars_sha256` in the col meta.

## Receipts

| Cell | Receipt | Rows | dip_xbeta finite |
|---|---|---|---|
| d1 (5×300) | `merge_d1.json` / `merge_d1.losses.npz` | 1500 | 1500 |
| h4f (5×300) | `merge_h4f.json` / `merge_h4f.losses.npz` | 1500 | 1500 |

Inference: d1 = 300 balanced aligned target times; h4f = 295 (10 excluded).
Protocol: origins=300, lookback=400, window=250, garch_window=750, seed=7,
n_boot=1000 — identical to prior merges.

## Results (pooled mean CRPS, 21 models per cell)

### d1 — rank 10/21, MCS @0.10 IN (p=0.364)

| Model | CRPS | | Model | CRPS |
|---|---|---|---|---|
| dip_fhs | 0.014945 | | dip_gmm_k | 0.015181 |
| dip_garch_t | 0.014981 | | dip_ewma_emp | 0.015192 |
| dip_regime | 0.015065 | | dip_ewma_t | 0.015293 |
| dip_qar | 0.015090 | | dip_gauss | 0.015438 |
| dip_conf_t | 0.015092 | | dip_lgbm_q | 0.015451 |
| dip_blend | 0.015095 | | **timesfm** | 0.015917 |
| dip_empirical | 0.015099 | | **chronos2** | 0.016390 |
| dip_student_t | 0.015103 | | **bolt_small** | 0.017857 |
| dip_stack | 0.015119 | | **kronos_small** | 0.064231 |
| **dip_xbeta** | **0.015134** | | dip_skt | 0.015146* |

*rank 11, listed for proximity. Δ vs dip_garch_t: +0.000153 (+1.0%, worse).
Δ vs best published target (timesfm): −0.000783 (−4.9%, better; DM p=0.0002).

### h4f — rank 14/21, MCS @0.10 OUT (p=0.023)

| Model | CRPS | | Model | CRPS |
|---|---|---|---|---|
| dip_fhs | 0.005218 | | dip_conf_t | 0.005327 |
| dip_garch_t | 0.005220 | | dip_student_t | 0.005329 |
| dip_stack | 0.005273 | | **dip_xbeta** | **0.005352** |
| dip_regime | 0.005284 | | dip_ewma_t | 0.005375 |
| dip_ewma_emp | 0.005306 | | dip_gauss | 0.005390 |
| dip_blend | 0.005312 | | dip_lgbm_q | 0.005475 |
| dip_gmm_k | 0.005324 | | **timesfm** | 0.005503 |
| dip_empirical_long | 0.005324 | | **bolt_small** | 0.005748 |
| dip_qar / dip_empirical | 0.005325 | | **chronos2** | 0.006035 |
| dip_skt | 0.005327 | | **kronos_small** | 0.007057 |

Δ vs dip_garch_t: +0.000132 (+2.5%, worse). Δ vs best published target
(timesfm): −0.000151 (−2.7%, better; DM p=0.0391).

### Per-asset dip_xbeta mean CRPS

| Asset | d1 | h4f |
|---|---|---|
| BNBUSDT | 0.012092 | 0.004414 |
| BTCUSDT | 0.012290 | 0.003636 |
| ETHUSDT | 0.016698 | 0.004909 |
| SOLUSDT | 0.017921 | 0.006177 |
| XRPUSDT | 0.016670 | 0.007625 |

## Verdict (honest)

dip_xbeta is a **mid-pack challenger**, not a SOTA-beating one. It beats every
published foundation target in both cells (d1: all four by ≥0.0008; h4f: all
four by ≥0.00015) but trails the leading univariate challengers — dip_fhs and
dip_garch_t — in both cells, by ~1% on d1 and ~2.5% on h4f. It sits inside the
d1 MCS@0.10 and just outside on h4f. The cross-asset beta conditioning adds
no measurable edge over own-history GARCH/FHS vol on this 5-asset panel: BTC
beta exposure largely replicates what the target's own EWMA vol already
captures. All numbers above are read directly from the two merge receipts;
no claims beyond them.

## Artifacts

- Column tool: `scripts/_xbeta_col.py` (standalone; imports helpers from
  `sota_eval_kronos` + `quant_fund` only; no existing file edited).
- Columns: `.dsh-24x7/lane-xbeta/<shard>.xbeta.npz` (10 files, computed on
  remote me@100.116.120.51 `D:\dipcatcher`; all 10 shard sha256 matched local
  before compute).
- Spliced shards (21 cols each): `.dsh-24x7/lane-xbeta/spliced/*.xbeta.npz`.
- Receipts: `.dsh-24x7/lane-xbeta/merge_d1.json` (+`.losses.npz`),
  `merge_h4f.json` (+`.losses.npz`).
