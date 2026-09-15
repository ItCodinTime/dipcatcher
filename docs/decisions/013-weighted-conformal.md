# ADR-013: Weighted split CQR under covariate shift

## Status

Accepted

## Date

2026-09-15

## Context

SYNTHETIC plants a vol-regime shift (idiosyncratic scale doubles after \(t/2\)). Exchangeable split CQR is valid only when calibration and test covariates are exchangeable. After the shift, cal is low-vol-heavy and test is high-vol-heavy, so a single \(\hat q\) is too tight on the test slice. Mondrian CQR (ADR-009) discretizes the same PIT-safe vol covariate. We still need the continuous likelihood-ratio version that matches the planted shift.

## Options considered

- Ignore the shift and keep exchangeable `SplitCQR` — undercovers the high-vol test window.
- Mondrian terciles of vol (ADR-009) — valid given a discrete stratum, not a density ratio.
- Weighted split CQR (Tibshirani, Barber, Candès, Ramdas 2019) with \(w(x)=dP_{\mathrm{test}}/dP_{\mathrm{cal}}\) estimated by a 1-d histogram on vol — distribution-free given the weights, no new learner.
- Neural density-ratio weights — blocked until conformal baselines exist (ADR-007).

## Decision

Implement `WeightedSplitCQR` in `quant_fund.models.weighted_conformal`. Weights are a Laplace-smoothed histogram ratio on a PIT-safe 1-d covariate (vol), clipped to a finite range. The weighted quantile is the analog of `conformal_quantile`:

\[
\tilde w = w\cdot n/\textstyle\sum w,\qquad
\ell=\min\bigl(1,\;\lceil(1-\alpha)(1+\textstyle\sum \tilde w)\rceil / \textstyle\sum \tilde w\bigr)
\]

then the ``higher`` weighted quantile of the CQR scores at level \(\ell\). This is the finite-sample \((1-\alpha)(1+\sum w)/\sum w\) analog of Tibshirani et al. (2019), Eq. (7), with the extra test atom taken as 1 after mean-1 rescaling. One \(\hat q\) per query histogram bin. Lab scores are coverage and width. No Sharpe. PIT of the raw model is unchanged.

## Consequences

The helper `bench_weighted_cqr` lives on the new module (existing benches are not edited). High-\(|Y|\) coverage stays a diagnostic only. Weight estimation error can still leak coverage; clip and Laplace smoothing bound the ratio.
