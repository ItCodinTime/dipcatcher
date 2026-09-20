# ADR-027: RP-PCA, FNW, three-pass, double-selection challengers

## Status

Accepted

## Date

2026-09-19

## Context

ADR-026 added VoC RFF, KNS SDF ridge, and restricted IPCA as catalog
names. They were not on the scientific bench and `forecast_asof` still
loaded only ridge. The user asked to go further: read more landmark
papers and implement them as causal public-feature engines.

NBER pages/PDFs used: w24858 (Lettau–Pelger), w23227 (FNW), w23527
(Giglio–Xiu). Feng–Giglio–Xiu JoF 2020 and KNS eq. 28 from the already
extracted w24070 text.

## Decision

1. Add catalog names `rff_ridgeless`, `sdf_en`, `ipca_alpha`, `rp_pca`,
   `fnw`, `gx3pass`, `ds_lasso`. Formulas and deviations in MATH_SPEC.
2. `train.paper_rankers` is the bench list. `bench_ranking` runs each on
   `PUBLIC_FEATURES` with the same purged walk-forward as ridge and
   stores date-level IC / RankIC / pairwise DM. Skip a name if fit
   raises; do not invent scores.
3. `forecast_asof` remains ridge (or the momentum heuristic if no ridge
   joblib). Paper joblibs, if present, are stamped (`paper_challengers`,
   Spearman vs ridge). Blend weight for these engines is identically 0.
4. No Sharpe, Sortino, Calmar, or P&L keys in ranker metadata or the
   research family blob. SYNTHETIC IC is a correctness test, not a
   promotion ticket.

## Consequences

- Research notebooks grow a larger ranker universe, so H99 data-snooping
  (SPA/MCS) is the honest multiple-testing layer.
- Default `paper_rankers` is the full set. Tiny CI panels
  (`n_dates < 80` or `n_names < 12`) skip the extra walk-forwards so
  unit benches stay fast. A serious panel always runs them. A name that
  cannot form managed portfolios is skipped, not scored as zero.
- Linear GKX autoencoder is IPCA; it is not a separate engine (ADR-007).
