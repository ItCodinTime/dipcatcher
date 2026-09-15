# ADR-020: Portfolio-level conformal sets for \(r_p = w^\top r\)

## Status

Accepted

## Date

2026-09-15

## Context

Name-level CQR / ACI (ADR-008, ADR-009) produce one interval per security. The book is a single scalar, \(r_p = w^\top r\). Coverage of per-name sets does not imply coverage of the book: names on the same date are dependent, and stacking name-date residuals treats a cross-section as if it were more i.i.d. dates. Interval caps (ADR-014) still consume name sets. The missing object is one conformal prediction set per date for the realized book return. Neural nets remain blocked (ADR-007). `optimizer.py` and `risk_gate.py` stay unchanged.

## Options considered

### Option A: Stack name-date CQR scores

- Pros: reuses `SplitCQR` with a larger sample.
- Cons: same-date names are not exchangeable dates. The nominal \(n\) is inflated and the coverage claim is not about \(r_p\).

### Option B: Interval arithmetic on name sets, \(\sum_i w_i[C_i]\)

- Pros: no new conformal fit.
- Cons: ignores residual dependence; the Minkowski sum is not a finite-sample \((1-\alpha)\) set for \(w^\top r\).

### Option C: Split CQR on the date-level scalar (chosen)

Walk dates. Each date contributes one realized \(r_p\) and one residual score. The base interval is a Gaussian band from historical (train-window) portfolio volatility. Optional portfolio `vol` is a PIT-safe scale for normalized CQR (ADR-009). Romano, Patterson, Candès (2019).

- Pros: the conformal sample matches the object we claim to cover; finite-sample \((1-\alpha)\) under exchangeable dates; no new learner.
- Cons: \(n\) is the number of dates, not names. Thin calendars need a longer burn-in.

### Option D: Per-name CQR plus Bonferroni / union

- Pros: valid for the event “every name is covered.”
- Cons: that event is not the book P&L; the union is far more conservative than a set for \(r_p\).

## Decision

Use **Option C** in `quant_fund.portfolio.portfolio_conformal`.

- `portfolio_return` is a nan-safe dot. Names with a non-finite weight or return are ignored; remaining weights are not renormalized.
- `SplitPortfolioCQR` calibrates CQR scores on a series of realized \(r_p\) against a base interval, then expands. Optional `vol` divides scores and multiplies the expansion (normalized CQR).
- `sets_by_date` walks dates in order, builds one \(r_p\) per date, fits the Gaussian band on a chronological train window, scores a held-out calibration window, and expands the test window. Calibration scores are never computed on test dates. Names are never stacked.
- `bench_portfolio_cqr` reports coverage, mean width, and \(n_{\mathrm{dates}}\) only. No Sharpe.

`risk_gate.py` and `optimizer.py` are not edited. Callers that need name caps still use ADR-014.

## Consequences

Lab coverage is a statement about the book path, not about the cross-section. A date with twenty names still adds one score. High name-level coverage can coexist with a book miss when residuals are correlated; that gap is now visible. Promotion still requires the raw Gaussian PIT on \(r_p\) to remain a specification diagnostic.
