# ADR-028: Fama–MacBeth, PCR/PLS, 3PRF, GBRT, principal portfolios

## Status

Accepted

## Date

2026-09-19

## Context

Wave 144 wired RP-PCA, FNW, Giglio–Xiu three-pass, and FGX double
selection. A causal SYNTHETIC public-feature walk-forward (16 names ×
120 days) showed IPCA slightly above ridge on date IC, while `fnw` and
`sdf_en` returned NaN IC because AG-LASSO / HJ-EN collapsed to constant
scores. The user asked to go further: read more landmark papers and
implement them as engines, not as live P&L.

Papers read: Fama–MacBeth (JPE 1973); Gu–Kelly–Xiu RFS 2020 / NBER
w25398 (PCR, PLS, GBRT+H; neural nets skipped); Kelly–Pruitt JoE 2015
3PRF (Penn working-paper PDF, Tables 1–2); Kelly–Malamud–Pedersen JoF
2023 / NBER w27388 (Wiley OA HTML + NBER PDF).

## Decision

1. Repair `fnw` (empty group-LASSO → full spline OLS) and `sdf_en`
   (zero loadings → \(\gamma_1=0\) path of KNS 28).
2. Add catalog names `fm`, `pcr`, `pls`, `tprf`, `gbrt`, `pp`. Formulas
   and deviations in MATH_SPEC.
3. `gbrt` is the GKX tree baseline (ADR-007). Neural nets and the
   nonlinear autoencoder stay unimplemented; linear CA remains IPCA.
4. `pp` needs security ids. `oos_rank_scores` / `train_ranking` /
   `forecast_asof` stamps pass them. Blend weight stays 0.
5. No Sharpe in metadata. SYNTHETIC IC is correctness, not promotion.

## Consequences

- The paper universe is 16 public-feature challengers. Tiny CI panels
  still skip it (`n_dates < 80` or `n_names < 12`).
- H99 SPA/MCS remains the multiple-testing layer on the larger horse
  race. Champion remains public ridge until a non-SYNTHETIC card wins.
