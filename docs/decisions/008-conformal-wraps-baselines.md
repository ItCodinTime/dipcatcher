# ADR-008: Conformal wraps baselines and does not relax PIT

## Status

Accepted

## Date

2026-09-15

## Context

Gaussian and historical intervals in the lab undercover (coverage 0.745 vs 0.90; Kupiec rejects VaR). Neural quantile nets are still blocked by ADR-007. We need finite-sample coverage without claiming the raw model is well specified.

## Decision

`SplitCQR` and `AdaptiveConformal` wrap existing quantile, Gaussian, EWMA, and tail predictors. They expand prediction sets from calibration residuals. They do **not** rewrite PIT, pinball, or CRPS of the raw model. Those remain the specification diagnostics. Conformal scores are coverage, width, and the ACI \(\alpha_t\) path. No Sharpe.

## Consequences

`quant research` reports a `conformal` family. ADR-007 still forbids default neural nets. PIT rejection of the Gaussian is expected and must stay visible.
