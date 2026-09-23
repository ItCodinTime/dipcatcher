# dip_egarch_l — TRUE asymmetric EGARCH(1,1)-t challenger (o=1)

## What it is

Leverage-channel challenger: per origin, an EGARCH fit via the shared
`_arch_fit` helper (`arch_model(rets*100, mean="Constant", vol="EGARCH", p=1,
o=1, q=1, dist="t", rescale=False)`, maxiter=300) on the same trailing
`garch_window=750` return window as `dip_garch_t`/`dip_egarch`. 1-step variance
forecast + fitted `nu`/`mu` are mapped to a location-scale Student-t (scale
`sigma*sqrt((nu-2)/nu)`), scored by the same closed-form `crps_student_t` and
pinball-at-TAUS path. Fully causal and deterministic.

This is the asymmetry order the `dip_egarch` lane never engaged: that variant's
spec chain was o=0-primary so all 3000 origins fitted the *symmetric* news
curve. Here **o=1 is primary**, so the `gamma[1]` leverage coefficient is
actually fitted.

Spec chain per origin (deterministic):
1. `o=1` fit; rejected if the fit raises (`n_o1_exception`) **or** the 1-step
   sigma forecast is degenerate (`n_o1_degenerate`): non-finite, <=0, or
   >1.0 (=100%/bar stdev — definitionally broken on the 1d/4h majors grid;
   observed sane forecasts <=~0.2, observed explosive ones 32…1e151).
2. Rejected o=1 falls back to `o=0` for that origin (same degeneracy test).
3. Double failure → honest NaN (`n_convergence_failures`).

## Provenance

- Column tool: `scripts/_egarchl_col.py` (new file; mirrors `_egarch_col.py`
  grid/binding discipline; `dip_egarch_l` added to `KNOWN_MODELS` in
  `scripts/splice_challenger_column.py` — the only shared-file touch).
- Compute: remote `me@100.116.120.51` (`D:\dipcatcher`, arch 8.0.0, py
  3.12.10) via `scripts/_egarchl_run.ps1`. All 10 target shards, all 10 bars
  files, and `sota_eval_kronos.py` SHA256-verified identical to local before
  dispatch; `_egarchl_col.py` itself hash-verified post-scp
  (39fcfefb12b3…0509).
- Columns: `.dsh-24x7/lane-egarchl/egarchl_<shard>` (10 files).
- Spliced shards: `.dsh-24x7/lane-egarchl/spliced/<shard>` (21 model columns).
- Receipts: `.dsh-24x7/lane-egarchl/merge_d1.json` (+`.losses.npz`),
  `.dsh-24x7/lane-egarchl/merge_h4f.json` (+`.losses.npz`); merge via
  `sota_eval_kronos.py --merge-parts --bars-root data/raw/sources`, seed=7,
  n_boot=1000, timestamps `reconstructed_from_hash_verified_bars`.

## Results (pooled mean CRPS, 1500 rows per panel)

### d1 (daily, 5 assets x 300 origins)

| rank | model           | pooled CRPS |
|------|-----------------|-------------|
| 1    | dip_fhs         | 0.014945    |
| 2    | **dip_egarch_l**| **0.014952**|
| 3    | dip_garch_t     | 0.014981    |
| 4    | dip_regime      | 0.015065    |
| 5    | dip_qar         | 0.015090    |

Best target (timesfm) = 0.015917 → every challenger beats every target.
DM timesfm vs dip_egarch_l: diff +0.000965, t=+7.006, p≈0.0000.

### h4f (4h, 5 assets x 300 origins)

| rank | model           | pooled CRPS |
|------|-----------------|-------------|
| 1    | dip_fhs         | 0.005218    |
| 2    | dip_garch_t     | 0.005220    |
| 3    | **dip_egarch_l**| **0.005243**|
| 4    | dip_stack       | 0.005273    |
| 5    | dip_regime      | 0.005284    |

Best target (timesfm) = 0.005503. DM timesfm vs dip_egarch_l: p≈0.0000.

### Delta vs dip_egarch (symmetric, o=0) — the headline

| panel | dip_egarch (o=0) | dip_egarch_l (o=1) | delta    | leverage helped? |
|-------|------------------|--------------------|----------|------------------|
| d1    | 0.014976         | 0.014952           | -0.000024| YES — new #2, still behind dip_fhs |
| h4f   | 0.005221         | 0.005243           | +0.000022| NO — slightly worse, same rank 3 |

Pinball detail (egarch_l vs symmetric egarch):
- d1: q05 0.003088 vs 0.003132 (win), q50 0.010007 vs 0.010007 (tie),
  q95 0.003547 vs 0.003585 (win). Also beats dip_garch_t pooled (0.014981).
- h4f: q05 0.001023 vs 0.001012, q50 0.003441 vs 0.003436, q95 0.001505 vs
  0.001485 — symmetric wins all three taus by ~1e-5 each.

### MCS @0.10 (Hansen-Lunde-Nason)

- d1: **IN**, p=0.776. Set: {student_t, empirical, garch_t, fhs, ewma_emp,
  blend, skt, qar, conf_t, regime, egarch_l}.
- h4f: **IN**, p=0.524. Set: {student_t, empirical, empirical_long, garch_t,
  fhs, ewma_emp, blend, gmm_k, skt, conf_t, regime, stack, egarch_l}.

## NaN / fallback accounting

- Finite CRPS: 1500/1500 (d1) and 1500/1500 (h4f); coverage 1.0 everywhere.
- o=1 scored **2997/3000** origins; o=0 fallback used on **3** origins, all in
  `h4f_ethusdt` — all via the degenerate-forecast path (`n_o1_degenerate=3`,
  `n_o1_exception=0`), where the o=1 recursion produced explosive variance
  forecasts (sigma up to ~1e151; fitted gamma down to -3.77e6). The same
  origins score ~0.0015 under o=0.
- Hard convergence failures (double failure → NaN): **0**.

## Gamma sign — did the leverage channel materialize?

Negative `gamma[1]` = vol rises more on negative returns (classical leverage).

| panel  | asset | gamma<0 count | gamma median |
|--------|-------|---------------|--------------|
| d1     | BTC   | 300/300       | -0.105       |
| d1     | ETH   | 300/300       | -0.124       |
| d1     | SOL   | 300/300       | -0.091       |
| d1     | BNB   | 272/300       | -0.043       |
| d1     | XRP   | 122/300       | +0.002       |
| **d1 pooled** | | **1294/1500 (86%)** |      |
| h4f    | BTC   | 300/300       | -0.061       |
| h4f    | ETH   | 296/297       | -0.052       |
| h4f    | SOL   | 153/300       | -0.0003      |
| h4f    | BNB   | 0/300         | +0.059       |
| h4f    | XRP   | 1/300         | +0.026       |
| **h4f pooled** | | **850/1497 (57%)** |     |

Leverage is real but regime/asset-dependent: strongly negative on daily bars
for BTC/ETH/SOL/BNB, and on 4h only for BTC/ETH. BNB and XRP at 4h fit a
reliably *positive* gamma (inverse leverage — vol responds more to up moves),
and XRP/SOL-4h hover near zero. That split is consistent with the scoreboard:
o=1 gains on d1 (where gamma<0 dominates) and loses slightly on h4f (where
half the panel fits no leverage or the wrong sign).

## Honest caveats

1. The 3 explosive o=1 fits on h4f_ethusdt were handled by the disclosed
   degeneracy bound (sigma>1.0 → o=0 fallback). Without it those rows carry
   finite-but-broken CRPS ~1e150 — the fallback is the "fits poorly" branch,
   not a silent fix; counts are in every column's meta.
2. Wins are small: -24e-6 (d1) / +22e-6 (h4f) vs symmetric `dip_egarch` —
   inside MCS noise on both panels; `dip_fhs` remains the top challenger.
3. `gamma` magnitude outliers exist among accepted fits (a few |gamma|>100
   with sane forecasts); medians above are robust, means are not reported
   headline for that reason.
4. No promotion claim: rank 2 (d1) / rank 3 (h4f), MCS member on both panels,
   zero NaN.

## Verdict

The true leverage channel is confirmed in the fits (gamma<0 on 86% of daily
origins) and it *helps* on the daily panel — `dip_egarch_l` takes rank 2/21,
edging both `dip_garch_t` and symmetric `dip_egarch`, beaten only by
`dip_fhs`. On 4h it is a slight net negative (rank 3, +22e-6 vs symmetric)
because half the 4h assets fit positive or ~zero gamma. Net: keep as a
diversity member — the leverage story is real on d1 but not a uniform win.
