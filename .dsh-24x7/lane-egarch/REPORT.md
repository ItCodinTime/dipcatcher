# dip_egarch — EGARCH(1,1)-t challenger

## What it is

Asymmetric-volatility challenger: per origin, an EGARCH fit via the shared
`_arch_fit` helper (`arch_model(rets*100, mean="Constant", vol="EGARCH", p=1,
q=1, dist="t", rescale=False)`, maxiter=300) on the same trailing
`garch_window=750` return window as `dip_garch_t`. 1-step variance forecast +
fitted `nu`/`mu` are mapped to a location-scale Student-t (scale
`sigma*sqrt((nu-2)/nu)`), scored by the same closed-form `crps_student_t` and
pinball-at-TAUS path as `dip_garch_t`. Fully causal and deterministic.

Spec chain per origin: `o=0` first, `o=1` on hard exception, honest NaN on
double failure or degenerate/non-finite variance forecast.

## Provenance

- Column tool: `scripts/_egarch_col.py` (new file; mirrors `_challenger_col.py`
  grid/binding discipline).
- Compute: remote `me@100.116.120.51` (`D:\dipcatcher`, arch 8.0.0, py 3.12.10).
  All 10 target shard SHA256 verified identical to local before dispatch.
- Columns: `.dsh-24x7/lane-egarch/egarch_<shard>` (10 files).
- Spliced shards: `.dsh-24x7/lane-egarch/spliced/<shard>` (21 model columns).
- Receipts: `.dsh-24x7/lane-egarch/merge_d1.json` (+`.losses.npz`),
  `.dsh-24x7/lane-egarch/merge_h4f.json` (+`.losses.npz`); merge via
  `sota_eval_kronos.py --merge-parts --bars-root data/raw/sources`, seed=7,
  n_boot=1000, timestamps `reconstructed_from_hash_verified_bars`.

## Results (pooled mean CRPS, 1500 rows per panel)

### d1 (daily, 5 assets x 300 origins)

| rank | model          | pooled CRPS |
|------|----------------|-------------|
| 1    | dip_fhs        | 0.014945    |
| 2    | **dip_egarch** | **0.014976**|
| 3    | dip_garch_t    | 0.014981    |
| 4    | dip_regime     | 0.015065    |
| 5    | dip_qar        | 0.015090    |

Best target (timesfm) = 0.015917 → every challenger beats every target.

### h4f (4h, 5 assets x 300 origins)

| rank | model          | pooled CRPS |
|------|----------------|-------------|
| 1    | dip_fhs        | 0.005218    |
| 2    | dip_garch_t    | 0.005220    |
| 3    | **dip_egarch** | **0.005221**|
| 4    | dip_stack      | 0.005273    |
| 5    | dip_regime     | 0.005284    |

Best target (timesfm) = 0.005503.

### Delta vs dip_garch_t

- d1: -0.000005 CRPS (egarch better; wins q05 0.003132 vs 0.003139 and q50
  0.010007 vs 0.010008, loses q95 0.003585 vs 0.003537).
- h4f: +0.000001 CRPS (a statistical tie; loses q05/q95 by ~4e-6/1.5e-6).

### MCS @0.10 (Hansen-Lunde-Nason)

- d1: **IN**, p=0.369. Set: {student_t, empirical, garch_t, fhs, ewma_emp,
  blend, skt, qar, conf_t, regime, egarch}.
- h4f: **IN**, p=0.909. Set: {student_t, empirical, empirical_long, garch_t,
  fhs, ewma_emp, blend, gmm_k, skt, conf_t, regime, stack, egarch}.

## NaN / failure accounting

- Finite CRPS: 1500/1500 (d1) and 1500/1500 (h4f); coverage 1.0 everywhere.
- Spec used: `egarch_t_o0` on all 3000 origins; `o=1` fallback never invoked;
  hard convergence failures = 0.

## Honest caveats

1. **Asymmetry note**: in `arch`, EGARCH's leverage (gamma) terms are
   controlled by `o`; the spec chain landed on `o=0` for every origin, so the
   fitted model is the *symmetric* EGARCH news curve (alpha on |e| only). The
   "asymmetric-vol challenger" framing is therefore only half-delivered —
   performance gains vs `dip_garch_t` come from the log-variance spec and the
   |e| innovation, not from a fitted leverage coefficient. A `o=1`-primary
   rerun would test the true leverage channel.
2. `dip_egarch` ≈ `dip_garch_t` within noise on both panels (h4f dead heat,
   d1 +5e-6 edge); it does not displace `dip_fhs` as the top challenger.
3. All DM tests vs the four targets are p<=0.0002 in egarch's favor, same as
   the rest of the challenger pack — the interesting comparison is inside the
   MCS, where it is mid-pack among vol challengers.

## Verdict

Solid but not dominant: rank 2 (d1) / rank 3 (h4f) overall, MCS member on
both panels, zero NaNs, cheaper-than-target accuracy. Incremental over
`dip_garch_t` on d1, a coin-flip on h4f. Keep as a diversity member; if a
leverage-specific claim is wanted, rerun with `o=1` primary.
