# dip_evt — EVT tail-augmented empirical challenger (lane-evt)

Date: 2026-09-23. Lane owner: subagent run. Model: `dip_evt`.

## Model

Per origin, on the trailing `garch_window` (=750) close-to-close returns:

1. Thresholds `u_lo = q0.10(r)`, `u_hi = q0.90(r)` (two-sided POT); exceedance
   sets `{u_lo - r : r < u_lo}` and `{r - u_hi : r > u_hi}`, nominal tail mass
   0.10 each.
2. GPD (Pickands/Balkema–de Haan) fit per tail by **method of moments**
   (closed form: `xi = 0.5*(1 - m^2/v)`, `beta = 0.5*m*(m^2/v + 1)`), xi
   clipped to [-0.5, 0.5]. MoM chosen over `scipy.stats.genpareto.fit` MLE:
   deterministic, O(n), no optimizer failure modes. Documented in meta
   `evt_spec`.
3. Quantile function: empirical for tau in [0.10, 0.90]; GPD tails anchored at
   the thresholds (continuous at the seam: q(0.10)=u_lo, q(0.90)=u_hi).
4. Vol rescale: quantiles * `clip(ewma_next_sigma(r)/std(r), 0.7, 1.4)`.

Scoring: CRPS via 512-point midpoint-quantile copy + `crps_empirical` (same
convention as dip_skt/dip_conf_t/dip_regime); pinball at TAUS=(0.05,0.5,0.95)
on the rescaled hybrid quantile function. Quantiles also emitted at LGBM_TAUS
(`q_lgbm` array in each col artifact). Fallback: either tail with <8
exceedances or unstable fit -> plain empirical quantiles, unscaled, same
scoring; disclosed via `fallback_col`/`n_fallback_*` meta fields.

Implementation: `scripts/_evt_col.py` (new file only; no existing files
edited). Column bind = bars_sha256 + config + row count, verified by
`splice_challenger_column.py` (invoked with a runtime `KNOWN_MODELS.add(
"dip_evt")` patch — the file itself was not modified; adding `dip_evt` to that
set permanently is a one-line follow-up for the maintainer).

## Provenance / determinism

- Computed on `me@100.116.120.51` (`D:\dipcatcher`, `.venv\Scripts\python.exe`)
  via `.dsh-24x7\lane-evt\run_evt.ps1`; all 10 target shards verified
  sha256-identical remote vs local before trusting remote compute.
- Determinism check: remote d1_btcusdt col mean_crps 0.012035 == local smoke
  `/tmp/evt_smoke.npz` 0.01203515478620956 (bitwise-identical pipeline).
- ~0.8 s/shard remote (300 origins each).

## Results (verified contract `native_shapes_timesfm_point_first.v2+spliced`)

Pooled CRPS, 21-model arena (4 published targets + 17 challengers), 5 assets x
300 origins = 1500 rows per merge; inference on aligned equal-weight
per-timestamp panel (d1: 300 times; h4f: 295 balanced times).

### Daily (`merge_d1.json`) — dip_evt rank 2/21

| rank | model | pooled CRPS |
|---:|---|---:|
| 1 | dip_fhs | 0.014945 |
| **2** | **dip_evt** | **0.014951** |
| 3 | dip_garch_t | 0.014981 |
| 4 | dip_regime | 0.015065 |
| 5 | dip_qar | 0.015090 |
| 8 | dip_empirical | 0.015099 |
| 12 | dip_empirical_long | 0.015150 |
| 18 | timesfm | 0.015917 |
| 19 | chronos2 | 0.016390 |
| 20 | bolt_small | 0.017857 |
| 21 | kronos_small | 0.064231 |

MCS @0.10: dip_evt **IN** (p=0.923). Superior set = {student_t, empirical,
garch_t, fhs, ewma_emp, blend, skt, qar, conf_t, regime, evt} — 11 challengers;
all 4 published targets excluded.

### 4h (`merge_h4f.json`) — dip_evt rank 3/21

| rank | model | pooled CRPS |
|---:|---|---:|
| 1 | dip_fhs | 0.005218 |
| 2 | dip_garch_t | 0.005220 |
| **3** | **dip_evt** | **0.005256** |
| 4 | dip_stack | 0.005273 |
| 5 | dip_regime | 0.005284 |
| 11 | dip_empirical | 0.005325 |
| 9 | dip_empirical_long | 0.005324 |
| 18 | timesfm | 0.005503 |
| 19 | bolt_small | 0.005748 |
| 20 | chronos2 | 0.006035 |
| 21 | kronos_small | 0.007057 |

MCS @0.10: dip_evt **IN** (p=0.358). Superior set = 13 challengers; all 4
published targets excluded.

## Headline deltas (paired DM on aligned panel; negative diff = evt better)

| horizon | vs | delta CRPS | t | p | verdict |
|---|---|---:|---:|---:|---|
| d1 | dip_empirical | -0.000148 (-0.98%) | -1.886 | 0.060 | better, marginal |
| d1 | dip_empirical_long | -0.000199 | -2.315 | 0.021 | better, significant |
| d1 | dip_garch_t | -0.000029 (-0.20%) | -0.521 | 0.603 | statistical tie |
| d1 | dip_fhs | +0.000006 (+0.04%) | +0.094 | 0.925 | dead tie |
| h4f | dip_empirical | -0.000071 (-1.33%) | -1.828 | 0.069 | better, marginal |
| h4f | dip_empirical_long | -0.000069 | -2.514 | 0.013 | better, significant |
| h4f | dip_garch_t | +0.000035 (+0.67%) | +1.090 | 0.277 | behind, n.s. |
| h4f | dip_fhs | +0.000038 (+0.73%) | +1.442 | 0.151 | behind, n.s. |

dip_evt beats dip_empirical on **all 10 assets** pointwise (per-asset means in
receipts). Tail pinball improves as designed — daily pinball@0.95 0.003635 vs
empirical 0.003830 (-5.1%), @0.05 0.003103 vs 0.003134; 4h @0.95 0.001551 vs
0.001675 (-7.4%), @0.05 0.001015 vs 0.001067 (-4.9%). The median pinball is
unchanged (0.010014 d1, ~0.003439 h4f) — the gain comes from the tails, which
is the intended mechanism.

## NaN / fallback counts

0 NaN CRPS rows, 0 empirical fallbacks across all 3000 origins — the
10%-threshold POT design yields ~75 exceedances per tail at garch_window=750,
well above the <8 fallback floor. Mean fitted xi: daily +0.00..+0.16
(heaviest: XRP +0.15/+0.16 lo/hi; SOL +0.08/+0.12); 4h mixed, including
negative lower-tail xi on BNB (-0.04), SOL (-0.18), XRP (-0.04) — bounded
lower tails at 4h. Mean vol-ratio applied 0.80–1.05 (clip [0.7,1.4] engaged at
both ends on BTC d1).

## Verdict (honest)

- `dip_evt` is a **genuine methodological upgrade over dip_empirical**:
  strictly better pooled CRPS at both horizons and on every asset, with the
  improvement concentrated at the scored tail quantiles — the stated rationale
  is confirmed by the pinball decomposition. DM vs dip_empirical is marginal
  (p~0.06) but vs the same-window `dip_empirical_long` it is significant
  (p=0.021 d1 / 0.013 h4f).
- It does **not** separate from the frontier tier: statistically tied with
  dip_fhs/dip_garch_t at both horizons (inside MCS at both). On daily it is
  the #2 model by point estimate; on 4h #3.
- No promotion claim: rank-2/3 point estimates within the MCS tier, deltas vs
  the leaders are not significant. Published targets remain excluded from the
  superior set at both horizons under the verified contract.

## Artifacts

- `scripts/_evt_col.py` — column generator (local + remote copy at
  `D:\dipcatcher\scripts\_evt_col.py`; remote runner
  `D:\dipcatcher\.dsh-24x7\lane-evt\run_evt.ps1`).
- `.dsh-24x7/lane-evt/cols/*.dip_evt.npz` — 10 column artifacts (crps_col,
  pin_cols, target_time_ns, q_taus, q_lgbm, xi cols, scale, fallback mask,
  meta).
- `.dsh-24x7/lane-evt/spliced/*.v2aug_evt.npz` — 10 spliced shards (21 cols).
- `.dsh-24x7/lane-evt/merge_d1.json` + `merge_d1.losses.npz`,
  `merge_h4f.json` + `merge_h4f.losses.npz` — merged receipts (seed 7,
  n_boot 2000, timestamps recovered via --bars-root data/raw/sources).
- `/tmp/evt_smoke.npz` — local BTC-daily smoke (300 rows, matches remote).
