# ADR-016: CV-minmax wraps K-fold residual scores at coverage 1−α

## Status

Accepted

## Date

2026-09-15

## Context

Jackknife+ (ADR-011) reuses a residual or quantile predictor with leave-one-out mean±z bands. Its finite-sample identity is \(\ge 1-2\alpha\). We want K-fold residual scores that do not refit a new learner, do not stack a panel across dates, and carry a **harder** finite-sample bound than Jackknife+. Neural nets stay blocked by ADR-007.

Barber, Candès, Ramdas, Tibshirani (2021) give two intervals on the same out-of-fold predictors \(\hat\mu_{-S_{k(i)}}\):

- CV+ (eq. 11, Theorem 4): \(q^\pm\) of \(\{\hat\mu_{-S_{k(i)}}(x)\pm R_i^{\mathrm{CV}}\}\). Coverage \(\ge 1-2\alpha-\varepsilon_n\). This is **not** \(1-\alpha\).
- Minmax (eq. 10, Theorem 3) on those predictors: \(\bigl[\min_i\hat\mu_{-S_{k(i)}}(x)-q^+\{R_i\},\;\max_i\hat\mu_{-S_{k(i)}}(x)+q^+\{R_i\}\bigr]\). Coverage \(\ge 1-\alpha\). CV+ is always contained in this set.

JAW (weighted CV+ ensemble; Tibshirani, Barber, Candès, Ramdas 2019 likelihood-ratio weights) keeps the plus quantile and is \(\ge 1-2\alpha\) under covariate shift. It does not recover \(1-\alpha\).

## Options considered

- Keep Jackknife+ only — valid at \(1-2\alpha\), \(n\) leave-one-out models.
- Naive K-fold residual quantile around a full-fit center — no finite-sample guarantee (same failure mode as naive jackknife).
- Classic CV+ (`aggregation="plus"`) — cheaper than LOO; Theorem 4 is still \(1-2\alpha\). Do not claim \(1-\alpha\).
- K-fold minmax on the same scores (`aggregation="minmax"`) — Theorem 3 identity \(\ge 1-\alpha\). Wider than plus.
- JAW — optional weighted plus for shift; identity \(1-2\alpha\), not \(1-\alpha\).

## Decision

`CVPlus` lives in `quant_fund.models.cv_plus` and does not edit `jackknife_plus.py` or `models/conformal.py`. The algorithm \(\mathcal{A}\) is the intercept-only mean±\(z_{1-\alpha/2}\) Gaussian band on residuals \(y-\hat\mu(x)\), fitted on each complement fold. CQR scores are \(s_i=\max(q^{\mathrm{lo}}_{-k(i)}-y_i,\,y_i-q^{\mathrm{hi}}_{-k(i)})\). Default aggregation is **minmax**, so the lab wrapper's finite-sample coverage is **\(\ge 1-\alpha\)**. We do not claim \(1-\alpha\) for `plus` or `jaw`.

`fit(y, pred)` or `fit(y, lower=..., upper=...)` (mid residual). Optional `dates` assign folds on unique timestamps in time order; rows that share a date share a fold and are never stacked into iid row folds. Lab scores stay coverage and width via `set_metrics`. No Sharpe.

## Consequences

`quant research` may report `cv_plus` next to Jackknife+ and split CQR. Promotion uses the \(1-\alpha\) identity only for minmax. Plus and JAW stay labeled \(1-2\alpha\). Date grouping is a leakage constraint, not a new theorem.
