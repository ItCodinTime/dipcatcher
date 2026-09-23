# dip_stack2 — second-generation causal quantile stack (lane-stack2)

Date: 2026-09-23. Lane owner: subagent run. Model: `dip_stack2`.

## Model

Same engine as `dip_stack` (`scripts/_challenger_col.py`): per origin, each
base emits its quantile vector at `LGBM_TAUS`; per quantile level a simplex
weight vector is fit by exponentiated-gradient descent on pinball loss over a
trailing buffer of past (forecast, realized) pairs (warmup 50 / buffer 150 /
iters 100 / eta 2.0); Vincentized per-level convex combination preserves
monotonicity; warmup origins predict at uniform weights; honest NaN when <2
finite bases; origins are banked only after scoring (never train on
themselves). Fully causal and deterministic.

`STACK2_BASES` (13) = the nine `dip_stack` bases (`dip_gauss`,
`dip_student_t`, `dip_ewma_emp`, `dip_empirical`, `dip_garch_t`, `dip_fhs`,
`dip_gmm_k`, `dip_conf_t`, `dip_regime`) **plus** four new-generation bases
vendored into `scripts/_stack2_col.py` (self-contained; no cross-lane
imports):

- `dip_egarch` — EGARCH(1,1)-t via `_arch_fit(rets*100, vol="EGARCH",
  dist="t", o=0)`, deterministic `o=1` fallback, Student-t quantiles with
  scale `sigma*sqrt((nu-2)/nu)` (same map as the `dip_garch_t` base).
- `dip_evt` — two-sided POT GPD tails (MoM, xi clipped [-0.5,0.5]) over an
  empirical body at q10/q90, vol-rescaled by
  `clip(ewma_next_sigma/std, 0.7, 1.4)`; plain empirical fallback.
- `dip_seas` — calendar-slot empirical quantiles shrunk toward pooled
  (pseudo-count 40, min slot obs 15); daily -> UTC day-of-week, 4h ->
  `hour//4`.
- `dip_volm` — volume-surprise-conditioned EWMA sigma with shrunk
  standardized-residual quantiles.

Both conditional files (`_seas_col.py`, `_volm_col.py`) existed with callable
per-origin quantile functions and were vendored.

## Provenance / determinism

- Column tool: `scripts/_stack2_col.py` (new file; mirrors `_challenger_col.py`
  `_stack_column`/`_stack_fit`/`_stack_base_quantiles` discipline). Only other
  file touched: `"dip_stack2"` appended to `KNOWN_MODELS` in
  `scripts/splice_challenger_column.py`.
- Compute: remote `me@100.116.120.51` (`D:\dipcatcher`, py 3.12.10) via
  `.dsh-24x7/lane-stack2/run_stack2*.ps1`. All 10 shards + 10 bars verified
  sha256-identical remote vs local before dispatch. ~123–209 s/shard remote
  (1578 s total, two concurrent streams); local smoke on d1_btcusdt
  0.0120571545 == remote 0.012057 (bitwise-identical pipeline).
- Columns: `.dsh-24x7/lane-stack2/cols/stack2_<shard>` (10 files).
- Spliced shards (21 cols each): `.dsh-24x7/lane-stack2/spliced/<shard>`.
- Receipts: `merge_d1.json` / `merge_h4f.json` (+`.losses.npz`), seed 7,
  n_boot 1000, timestamps `reconstructed_from_hash_verified_bars`, contract
  `native_shapes_timesfm_point_first.v2+spliced`.
- NaN accounting: 300/300 finite CRPS on every shard (3000/3000 origins);
  every base finite on every origin; 300/300 origins banked per shard.

## Results (pooled mean CRPS, 1500 rows per panel, 21-model arena)

### d1 (daily) — dip_stack2 rank 3/21

| rank | model         | pooled CRPS | MCS @0.10        |
|---:|---|---:|---|
| 1 | dip_fhs        | 0.014945 | IN (p=1.000) |
| 2 | dip_garch_t    | 0.014981 | IN (p=0.313) |
| **3** | **dip_stack2** | **0.015042** | **IN (p=0.286)** |
| 4 | dip_regime     | 0.015065 | IN |
| 5 | dip_qar        | 0.015090 | IN |
| 10 | dip_stack     | 0.015119 | **OUT (p=0.001)** |
| 18 | timesfm       | 0.015917 | OUT |

MCS set (10): {student_t, empirical, garch_t, fhs, ewma_emp, blend, qar,
conf_t, regime, **stack2**} — all four published targets excluded.

### h4f (4h) — dip_stack2 rank 3/21

| rank | model         | pooled CRPS | MCS @0.10        |
|---:|---|---:|---|
| 1 | dip_fhs        | 0.005218 | IN (p=1.000) |
| 2 | dip_garch_t    | 0.005220 | IN (p=0.846) |
| **3** | **dip_stack2** | **0.005246** | **IN (p=0.313)** |
| 4 | dip_stack      | 0.005273 | **OUT (p=0.058)** |
| 5 | dip_regime     | 0.005284 | IN |
| 18 | timesfm       | 0.005503 | OUT |

MCS set (12): {student_t, empirical, empirical_long, garch_t, fhs, ewma_emp,
blend, gmm_k, skt, conf_t, regime, **stack2**} — targets excluded.

### Paired DM on the aligned panel (negative diff = stack2 better)

| horizon | vs | delta CRPS | t | p | verdict |
|---|---|---:|---:|---:|---|
| d1 | dip_stack   | -0.000077 (-0.51%) | -4.539 | <0.0001 | **better, significant** |
| d1 | dip_fhs     | +0.000097 (+0.65%) | +2.105 | 0.036 | behind, marginal |
| d1 | dip_garch_t | +0.000061 (+0.41%) | +1.556 | 0.121 | behind, n.s. |
| d1 | dip_regime  | -0.000022 | -0.903 | 0.367 | tie |
| h4f | dip_stack   | -0.000027 (-0.50%) | -3.265 | 0.0012 | **better, significant** |
| h4f | dip_fhs     | +0.000029 (+0.56%) | +1.630 | 0.104 | behind, n.s. |
| h4f | dip_garch_t | +0.000027 (+0.52%) | +1.112 | 0.267 | behind, n.s. |
| h4f | dip_regime  | -0.000039 | -1.968 | 0.050 | better, marginal |

`dip_stack2` beats `dip_stack` **pointwise on all 10 cells** (5/5 assets both
panels). Pinball: d1 q05 0.003122 vs stack 0.003160 (-1.2%), q95 0.003651 vs
0.003712 (-1.6%), q50 ~unchanged; h4f q05 0.001001 vs 0.001020 (-1.8%), q95
0.001525 vs 0.001555 (-1.9%). The gain concentrates in the tails, consistent
with adding the tail-focused `dip_evt`/`dip_egarch` bases.

## Stack-weight diagnostics (which bases drive it)

Mean effective stack weight per base, averaged over scored origins and taus,
then over the 10 cells (uniform = 1/13 = 0.0769):

| base | mean w | min–max across cells |
|---|---:|---|
| dip_gauss     | 0.0788 | 0.0775–0.0833 |
| dip_gmm_k     | 0.0777 | 0.0773–0.0786 |
| dip_student_t | 0.0776 | 0.0771–0.0786 |
| dip_empirical | 0.0776 | 0.0771–0.0786 |
| dip_regime    | 0.0772 | 0.0767–0.0780 |
| dip_conf_t    | 0.0771 | 0.0759–0.0787 |
| dip_ewma_emp  | 0.0766 | 0.0750–0.0775 |
| dip_seas      | 0.0767 | 0.0753–0.0780 |
| dip_garch_t   | 0.0764 | 0.0748–0.0771 |
| dip_evt       | 0.0764 | 0.0750–0.0774 |
| dip_egarch    | 0.0763 | 0.0752–0.0772 |
| dip_fhs       | 0.0762 | 0.0752–0.0770 |
| dip_volm      | 0.0754 | 0.0723–0.0767 |

Read: weights stay **near-uniform**. EG descent (eta 2.0, 100 iters) moves
little when base pinballs sit within ~1e-4 of each other, so `dip_stack2` is
essentially a Vincentized average of 13 diverse quantile functions — the gain
over `dip_stack` comes from the enlarged/stronger base set (diversification),
not from weight concentration. The tilts that do exist are sensible: at the
0.95 tail `dip_volm` is downweighted (0.072) and `dip_gauss` upweighted
(0.081); `dip_fhs`/`dip_garch_t`/`dip_egarch` are slightly discounted at both
tails (~0.075) since their correlated t-quantiles dominate the
already-represented vol family. Per-cell meta records `mean_weight`,
`mean_weight_by_tau`, `mean_weight_when_finite`, and `base_finite_origins`.

## Honest caveats

1. Rank-3 point estimates on both panels but **still behind the vol leaders**:
   `dip_fhs` is significantly better on d1 (DM p=0.036); `dip_garch_t` leads
   on both panels n.s. The stack does not displace the frontier — it narrows
   the gap and enters the MCS on both panels where `dip_stack` fell out on
   both (p=0.001 d1 / p=0.058 h4f on these merged panels).
2. The upgrade is additive, not transformative: -0.51%/-0.50% CRPS vs
   `dip_stack`, driven by base-set breadth since fitted weights ≈ uniform.
3. MCS membership is bootstrap-dependent; at different seeds `dip_stack`'s
   h4f p=0.058 sits near the 0.10 boundary. No promotion claim — treat
   `dip_stack2` as the strictly-better stack and a diversity-preserving
   challenger inside the MCS tier, not a new leader.

## Artifacts

- `scripts/_stack2_col.py` — column generator (remote copy at
  `D:\dipcatcher\scripts\_stack2_col.py`; runners
  `.dsh-24x7/lane-stack2/run_stack2*.ps1`).
- `.dsh-24x7/lane-stack2/cols/stack2_*.npz` — 10 column artifacts.
- `.dsh-24x7/lane-stack2/spliced/*.v2aug.npz` — 10 spliced shards (21 cols).
- `.dsh-24x7/lane-stack2/merge_{d1,h4f}.json` + `.losses.npz` — receipts.
- `/tmp/stack2_smoke.npz` — local BTC-daily smoke (300 rows, matches remote).
