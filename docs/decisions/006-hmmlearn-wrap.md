# ADR-006: Wrap hmmlearn despite limited maintenance

## Status

Accepted

## Date

2026-09-15

## Context

Need probabilistic regime inference. hmmlearn provides `GaussianHMM` and is specified. It is in limited-maintenance mode.

## Decision

Wrap `hmmlearn.hmm.GaussianHMM`. Do not impose state labels during fit. Interpret states after fit from mean return/vol/correlation. Isolate the import behind `quant_fund.models.regime.hmm`.

## Consequences

If hmmlearn breaks on a future Python, replace the wrapper. BIC/AIC/OOS likelihood choose \(k\), not portfolio Sharpe.
