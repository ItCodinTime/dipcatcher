# lane-lgbmqv — `dip_lgbm_qv` challenger evaluation

Model: `dip_lgbm_qv` — feature-augmented LightGBM quantile regression.
Identical trainer to `dip_lgbm_q` (`lgbm_quantiles` in
`scripts/sota_eval_kronos.py`: 9 quantile objectives at `LGBM_TAUS`,
`n_estimators=150, lr=0.06, num_leaves=15, min_child_samples=30,
subsample=0.9, random_state=7, n_jobs=2`, **per-origin refit**, monotonicity
via `np.maximum.accumulate`), but the causal feature table is the 8-column
`_lgbm_features` output plus 5 microstructure columns computed from bars
`<=` origin:

`park_vol` (sqrt mean Parkinson var, 20 bars), `rng_ratio` (last-bar
(high-low)/close), `vol_surp` (last volume / trailing median volume 20),
`amihud` (mean |r|/max(v,1e-12), 20 bars), `gap` (open/close_prev - 1).

Column tool: `scripts/_lgbmqv_col.py` (new file only; no existing files
edited). Column artifacts bind via `bars_sha256` + protocol fields; spliced
with `scripts/splice_challenger_column.py` (model already registered in
KNOWN_MODELS).

## Provenance / determinism

- Compute host: remote `me@100.116.120.51` (`D:\dipcatcher`,
  `.venv\Scripts\python.exe`, lightgbm 4.7.0). All 10 shard `.npz` inputs and
  all 10 bound bars verified sha256-identical local vs remote before launch;
  `scripts/sota_eval_kronos.py` identical (sha256 da47894c…).
- 10 parallel workers spawned via `Invoke-CimMethod Win32_Process Create`
  (absolute paths; relative-path first attempt wrote no logs and died —
  respawned cleanly, no partial outputs kept).
- Local smoke on `d1_btcusdt_1d_deep` (lightgbm 4.7.0, macOS) vs remote
  output: `target_time_ns` bit-identical; CRPS max |Δ| = 6.9e-18, pinball
  max |Δ| = 1.9e-17 (BLAS reduction-order noise only).

## Coverage / honesty

- Rows: 300/shard × 10 shards = 3000. Finite CRPS: **3000/3000**.
- Fit failures (`n_fit_failures` in each column meta): **0** on all shards.
- Pinball columns: 9000/9000 finite, all >= 0.
- NaN rows: none.

## Results (pooled mean CRPS, 5 assets × 300 origins = 1500 rows each)

### Daily (`merge_d1.json`)

| rank | model | CRPS |
|---:|---|---:|
| 1 | dip_fhs | 0.01494510 |
| 2 | dip_garch_t | 0.01498096 |
| 3 | dip_regime | 0.01506470 |
| 15 | **dip_lgbm_qv** | **0.01542485** |
| 16 | dip_gauss | 0.01543818 |
| 17 | dip_lgbm_q | 0.01545073 |
| 18 | timesfm | 0.01591689 |
| 21 | kronos_small | 0.06423109 |

- Δ vs `dip_lgbm_q`: **−0.0000259 (−0.17%)** — slightly better.
- Pinball pooled qv vs q: τ0.05 0.003125 vs 0.003195 (better);
  τ0.5 0.010500 vs 0.010563 (better); τ0.95 0.003758 vs 0.003705 (worse).
- MCS @0.10: `{dip_blend, dip_conf_t, dip_empirical, dip_ewma_emp, dip_fhs,
  dip_garch_t, dip_qar, dip_regime, dip_skt, dip_student_t}` —
  `dip_lgbm_qv` **out** (p=0.0040), `dip_lgbm_q` out (p=0.0020).
- Beats every published target (weakest DM: vs timesfm p=0.0132).

### 4h (`merge_h4f.json`)

| rank | model | CRPS |
|---:|---|---:|
| 1 | dip_fhs | 0.00521819 |
| 2 | dip_garch_t | 0.00522029 |
| 3 | dip_stack | 0.00527265 |
| 16 | dip_lgbm_q | 0.00547517 |
| 17 | **dip_lgbm_qv** | **0.00548029** |
| 18 | timesfm | 0.00550282 |
| 21 | kronos_small | 0.00705693 |

- Δ vs `dip_lgbm_q`: **+0.0000051 (+0.09%)** — slightly worse.
- Pinball pooled qv vs q: τ0.05 0.001087 vs 0.001052 (worse);
  τ0.5 0.003567 vs 0.003568 (~equal); τ0.95 0.001666 vs 0.001724 (better).
- MCS @0.10: `{dip_blend, dip_conf_t, dip_empirical, dip_empirical_long,
  dip_ewma_emp, dip_fhs, dip_garch_t, dip_gmm_k, dip_regime, dip_skt,
  dip_stack, dip_student_t}` — `dip_lgbm_qv` **out** (p=0.0010),
  `dip_lgbm_q` out (p=0.0010). (Panel composition changed: dip_ewma_t and
  dip_qar dropped out vs the 20-model v2aug merge.)
- Beats every published target (weakest DM: vs timesfm p=0.6616).

## Verdict

**Microstructure features are a wash.** `dip_lgbm_qv` tracks `dip_lgbm_q`
within ±0.2% pooled CRPS on both cells (better on d1, worse on h4f,
mixed sign per pinball level), stays mid-pack (rank 15–17 of 21), remains
outside the MCS at both horizons, and does not change the headline
ordering — `dip_fhs`/`dip_garch_t` still lead; all challengers still beat
every published target. The added columns are honest and causal (0
failures, 3000/3000 finite) but bought nothing on this protocol.

## Artifacts

- Columns: `.dsh-24x7/lane-lgbmqv/cols/{shard}.lgbmqv.npz` (+ `.log`) ×10
- Spliced shards: `.dsh-24x7/lane-lgbmqv/spliced/{shard}.v2aug.npz` ×10
  (21 columns each)
- Receipts: `.dsh-24x7/lane-lgbmqv/merge_d1.json`,
  `.dsh-24x7/lane-lgbmqv/merge_h4f.json` (+ matching `.losses.npz`)
- Tool: `scripts/_lgbmqv_col.py` (sha256 57e2088b…)
- Remote copies retained: `D:\dipcatcher\.dsh-24x7\lane-lgbmqv\`
