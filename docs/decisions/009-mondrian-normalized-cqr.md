# ADR-009: Mondrian and normalized CQR wrap baselines

## Status

Accepted

## Date

2026-09-15

## Context

Global CQR/ACI hit marginal 90% coverage on SYNTHETIC holdout, but coverage given large \(|Y|\) was ~70%. That slice is not a conformal guarantee: it selects the outcomes most likely to miss. Neural nets remain blocked by ADR-007. We still need locally adaptive sets that are valid given PIT-safe covariates.

## Options considered

- Localized neural quantile nets — forbidden until conformal baselines exist (ADR-007).
- Mondrian / normalized CQR wrapping existing Gaussian and linear QR — distribution-free, finite-sample, no new learner.
- Evaluating coverage given \(|Y|\) as a promotion gate — statistically invalid.

## Decision

`NormalizedCQR` scores are residual / `vol_20` (else predicted width). `MondrianCQR` and `MondrianACI` keep a separate \(\hat q\) and \(\alpha_t\) per calibration tercile of that scale. H8 tests the high-vol **X** slice, not \(|Y|\). PIT of the raw model is unchanged.

## Consequences

`quant research` reports `normalized_aci` and `mondrian_aci`. High-\(|Y|\) coverage stays a diagnostic only.
