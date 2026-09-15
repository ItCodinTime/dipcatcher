# ADR-017: Localized kernel-weighted CQR on PIT-safe vol

## Status

Accepted

## Date

2026-09-15

## Context

Global split CQR (ADR-008) uses one residual quantile. Mondrian CQR (ADR-009) splits that quantile across discrete vol terciles. Weighted split CQR (ADR-013) reweights the *same* calibration scores by a likelihood ratio \(dP_{\mathrm{test}}/dP_{\mathrm{cal}}\) under covariate shift. None of those give a *smooth* local residual law \(s\mid x\) on a continuous PIT-safe covariate. Neural localizers stay blocked by ADR-007.

## Options considered

- Mondrian bins of vol (ADR-009) — valid given a stratum, discontinuous at cutpoints, needs a minimum count per bin.
- Tibshirani likelihood-ratio weights (ADR-013) — correct for a global shift, not for heteroskedasticity at a query \(x\).
- Localized / kernel-weighted residual scores (Lei and Wasserman 2014; Guan 2023) with \(w_i(x)=\exp(-((x_i-x)/h)^2)\) — continuous in \(x\), no new learner, same CQR scores as the other wrappers.
- Localized neural quantile nets — forbidden until conformal baselines exist (ADR-007).

## Decision

Implement `LocalizedCQR` in `quant_fund.models.localized_conformal`. The covariate is 1-d PIT-safe vol. Bandwidth \(h\) is the calibration median of positive pairwise \(|x_i-x_j|\), or a passed value. For each unique query \(x\),

\[
w_i(x)=\mathrm{clip}\bigl(\exp\bigl(-((x_i-x)/h)^2\bigr)\bigr),\qquad
n_{\mathrm{eff}}=\bigl(\textstyle\sum w\bigr)^2/\textstyle\sum w^2.
\]

If \(n_{\mathrm{eff}}\) is below `min_ess` (default 12), use the global `conformal_quantile`. Otherwise take the finite-sample weighted conformal quantile of the CQR scores (same \((1+\sum w)\) level as ADR-013). Expand with `expand_interval`. Lab scores are coverage and width. No Sharpe. PIT of the raw model is unchanged.

We do **not** run Guan’s nested \(\tilde\alpha(x)\) search. Sparse kernels fall back to the exchangeable global quantile instead of emitting infinite sets.

## Consequences

High-vol queries get wider sets than low-vol queries when residuals scale with vol. Homoskedastic (constant-\(x\)) widths match global CQR. Marginal coverage on exchangeable data is the lab gate (\(\ge 0.85\) at \(\alpha=0.10\)), not a local-coverage theorem. Complementary to ADR-013: that module shifts the *sample*, this one shifts the *neighborhood*.
