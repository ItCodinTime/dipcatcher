# ADR-011: Jackknife+ wraps residual predictors at coverage 1-2α

## Status

Accepted

## Date

2026-09-15

## Context

Split CQR (ADR-008) needs a held-out calibration window and a single residual quantile. Naive jackknife, \(\hat\mu(x)\pm q^+\{R_i^{\mathrm{LOO}}\}\), has **no** finite-sample coverage guarantee and can fail when the fit is unstable. We want leave-one-out conformal intervals that reuse an existing residual or quantile predictor and do not reimplement QR. Neural nets stay blocked by ADR-007.

## Options considered

- Naive jackknife (full-fit \(\hat\mu\) plus LOO residual quantile) — cheap, but coverage can vanish.
- Split CQR / ACI — valid at \(1-\alpha\), already shipped; wastes a holdout and is not LOO.
- Jackknife+ (Barber, Candès, Ramdas, Tibshirani 2021) on an exact LOO mean±z band with CQR residual scores — distribution-free coverage \(\ge 1-2\alpha\), \(O(n)\) closed form, no new learner.
- JAW or CV+ — can recover \(1-\alpha\) under extra conditions; not implemented here. Do not claim \(1-\alpha\).

## Decision

`JackknifePlus` lives in `quant_fund.models.jackknife_plus` and does not edit `models/conformal.py`. The algorithm \(\mathcal{A}\) is the intercept-only mean±\(z_{1-\alpha/2}\) Gaussian band on residuals \(y-\hat\mu(x)\), with exact leave-one-out location and scale. CQR scores are \(s_i=\max(q^{\mathrm{lo}}_{-i}-y_i,\,y_i-q^{\mathrm{hi}}_{-i})\). Prediction uses the Jackknife+ ensemble

\[
\bigl[q^-_{n,\alpha}\{\hat\mu_{-i}(x)-s_i\},\;q^+_{n,\alpha}\{\hat\mu_{-i}(x)+s_i\}\bigr],
\]

not a single full-fit center. `fit_residuals` takes existing bands, sets \(\hat\mu=\) mid, and applies the same residual jackknife (\(R_i=|y_i-\mathrm{mid}_i|\cdot n/(n-1)\) when the residual mean is zero). Finite-sample coverage is **\(\ge 1-2\alpha\)**. We do not claim \(1-\alpha\).

## Consequences

Lab scores stay coverage and width via `set_metrics`. No Sharpe. `quant research` may report `jackknife_plus` next to split CQR; promotion still uses the \(1-2\alpha\) identity until JAW/CV+ exists.
