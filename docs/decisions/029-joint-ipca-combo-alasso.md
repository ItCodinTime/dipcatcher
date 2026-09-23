# ADR-029: Joint unrestricted IPCA, combination, adaptive LASSO

## Status

Accepted

## Date

2026-09-20

## Context

Wave 145 added Fama–MacBeth, PCR/PLS, 3PRF, GBRT, and principal
portfolios. Unrestricted IPCA (`ipca_alpha`) was still a nested one-shot
residual projection of \(r-Z\Gamma f\) onto \(Z\), which is not the
Kelly–Pruitt–Su unrestricted object. The non-SYNTHETIC gate remains a
public-feature win vs ridge on the Yahoo session-close tape. Overlay
Sharpe is cash-scaled, not alpha.

## Decision

1. Replace nested \(\Gamma_\alpha\) with joint ALS:
   \(F_{\mathrm{aug},t}=(1,f_t)'\), packed \(\Gamma_{\mathrm{aug}}=[\Gamma_\alpha\mid\Gamma]\),
   restricted warm-start, identification on \(\Gamma\) only.
   Formulas and deviations in MATH_SPEC.
2. Add catalog names `combo` (Rapach–Strauss–Zhou equal-weight univariate
   OLS) and `alasso` (Zou 2006 adaptive LASSO). `alasso_alpha` is the
   \(\ell_1\) penalty on reweighted columns.
3. Champion remains public ridge. `blend_weight` 0.
   `paper_challengers_blend` 0. `live_pnl_claim` false. No Sharpe in
   ranker metadata. SYNTHETIC IC is correctness, not promotion.

## Consequences

- The paper universe is 18 public-feature challengers. Tiny CI panels
  still skip it (`n_dates < 80` or `n_names < 12`).
- File-tape date IC vs public ridge is the promotion gate, not overlay
  Sharpe / Calmar / Sortino and not SYNTHETIC IPCA.
- Walk-forward row masks use integer-nanosecond membership. Object
  ``np.isin`` cannot finish a 100k-row expanding horse race.
- Yahoo session-close tape (54 names, 1134 OOS dates): RP-PCA mean
  date IC 0.020 (t=2.42) vs ridge 0.012 (t=1.51). Pairwise DM on −IC
  does not reject equal accuracy (p=0.14). That does not blend the
  book. `forecast_asof` still loads ridge.
