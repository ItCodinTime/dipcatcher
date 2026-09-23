# Lane `har` — `dip_har` challenger report

**Model:** `dip_har` — HAR (Heterogeneous AutoRegressive, Corsi 2009) realized-volatility
challenger on Parkinson range variance, scored by filtered historical simulation.

**Mechanism (new to the arena):** every other vol challenger uses close-close returns
only. `dip_har` builds a per-bar realized-variance series from Parkinson range variance
`pk_t = (ln high_t − ln low_t)² / (4 ln 2)` (≈5× more efficient than `r_t²`; per-bar
fallback to `r_t²` if non-finite — never triggered: 2,250,000/2,250,000 bars used
Parkinson across the 10 shards). A HAR regression in vol form,
`sv_{t+1} ~ c + b1·sv_t + b2·mean(sv_{t−4..t}) + b3·mean(sv_{t−21..t})` on `sv = √RV`
(OLS on the trailing `garch_window`, ≥60 regression rows, coefficients clipped ≥0),
forecasts `sigma_hat = max(pred, 1e-6)`. Quantiles are FHS-style:
`q_τ = μ_hat + sigma_hat · quantile(z, τ)` over `z_t = r_t/√RV_t` (≥30 finite z
required), `μ_hat = 0.5·mean(rets_long)`. CRPS via `crps_from_quantiles(y, LGBM_TAUS, q)`.
Fully deterministic, causal, ~0.26 s/shard.

## Compute provenance

- Script: `scripts/_har_col.py` (new; mirrors `_challenger_col.py` grid/binding
  discipline — `bars_sha256` + protocol fields + asset identity checked by
  `splice_challenger_column.py` at splice time).
- Computed remote-first on `me@100.116.120.51` (`D:\dipcatcher`) via
  `scripts/_run_har.ps1`. All 10 target shard SHA-256s verified identical
  local↔remote before compute; `sota_eval_kronos.py` and `sota_evidence.py`
  file hashes identical both sides (`pinball_loss` byte-identical despite an
  unrelated `scoring.py` file-level diff). Remote btc smoke reproduced the local
  smoke bit-for-bit (mean_crps 0.012209).
- Columns: `.dsh-24x7/lane-har/<shard>.harcol.npz` (10 files)
- Spliced shards: `.dsh-24x7/lane-har/spliced/<shard>.har.npz` (21 columns each)
- Merges: `.dsh-24x7/lane-har/merge_d1.json` / `merge_h4f.json` (+ `.losses.npz`),
  `--bars-root data/raw/sources`, timestamps `reconstructed_from_hash_verified_bars`,
  inference_seed 7, n_boot 1000 — same convention as `merge_d1_v2aug.json`.

## Results — pooled mean CRPS (1500 rows/lane, 1500 complete, 0 NaN for dip_har)

### Daily lane (`merge_d1.json`, n_obs=300 aligned)

| rank | model | pooled CRPS |
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
| **12** | **dip_har** | **0.015151** |
| 13 | dip_gmm_k | 0.015181 |
| 14 | dip_ewma_emp | 0.015192 |
| 15 | dip_ewma_t | 0.015293 |
| 16 | dip_gauss | 0.015438 |
| 17 | dip_lgbm_q | 0.015451 |
| 18 | timesfm | 0.015917 |
| 19 | chronos2 | 0.016390 |
| 20 | bolt_small | 0.017857 |
| 21 | kronos_small | 0.064231 |

### 4h lane (`merge_h4f.json`, n_obs=295 aligned)

| rank | model | pooled CRPS |
|---:|---|---:|
| 1 | dip_fhs | 0.005218 |
| 2 | dip_garch_t | 0.005220 |
| 3 | dip_stack | 0.005273 |
| **4** | **dip_har** | **0.005283** |
| 5 | dip_regime | 0.005284 |
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

## Key numbers

| metric | d1 | h4f |
|---|---|---|
| dip_har pooled CRPS | 0.015151 | 0.005283 |
| rank | 12/21 | 4/21 |
| Δ vs dip_garch_t | +0.000170 | +0.000062 |
| Δ vs dip_fhs (leader) | +0.000206 | +0.000064 |
| MCS @0.10 | **excluded** (p=0.021) | **excluded** (p=0.002) |
| NaN rows | 0/1500 | 0/1500 |
| DM vs kronos_small | +0.049080, t=11.59, p<0.0001 (har preferred) | +0.001777, t=3.67, p=0.0003 (har preferred) |
| DM vs chronos2 | +0.001239, p<0.0001 (har) | +0.000760, p=0.0001 (har) |
| DM vs bolt_small | +0.002705, p<0.0001 (har) | +0.000460, p<0.0001 (har) |
| DM vs timesfm | +0.000766, p<0.0001 (har) | +0.000221, p<0.0001 (har) |

Per-asset dip_har CRPS — d1: BNB 0.012071 / BTC 0.012209 / ETH 0.016862 /
SOL 0.017911 / XRP 0.016703. h4f: BNB 0.004271 / BTC 0.003680 / ETH 0.004971 /
SOL 0.006020 / XRP 0.007471.

Fitted HAR structure (per-origin mean coefs): d1 assets b1≈0.25–0.29,
b5≈0.18–0.27, b22≈0.02–0.27, c≈0.008–0.017; h4f shifts weight to the 22-bar
component (b1≈0.11–0.26, b5≈0.21–0.40, b22≈0.15–0.38, c≈0.001–0.003) —
consistent with 22 4h bars ≈ 3.7 days acting as the "monthly" horizon.

## Verdict — honest

`dip_har` is a legitimate mid-pack challenger, not a leader. On daily bars it
lands 12/21, +1.1% CRPS behind `dip_garch_t` and +1.4% behind `dip_fhs`. On 4h
it ranks 4/21 by pooled mean — but is still excluded from the MCS at p=0.002,
i.e. its loss process is dominated by the leaders with high confidence, not by
noise. It beats all four deep targets on every lane with decisive DM statistics,
and it does so with a genuinely different mechanism (range-based RV + HAR
persistence + empirical residual shape). The Parkinson edge does not overcome
the incumbent GARCH-t/FHS vol path on 1-step close-to-close returns — the extra
intraday range information buys roughly nothing once the leader already filters
vol well, and on d1 the hybrid (Parkinson RV feeding a return-space FHS) is
slightly worse than plain historical returns.
