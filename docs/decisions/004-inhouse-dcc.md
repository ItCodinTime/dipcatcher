# ADR-004: In-house DCC(1,1) on arch univariates

## Status

Accepted

## Date

2026-09-15

## Context

`arch` is univariate. `pymgarch` is 0.1.x. R `rmgarch` is the academic reference but is not a Python dependency we want.

## Decision

Implement Gaussian DCC(1,1) two-stage QML in `quant_fund.models.covariance.dcc`, wrapping `arch` for stage 1. Test \(a,b>0\), \(a+b<1\), PSD of \(R_t\), and recovery on a synthetic CCC/DCC process.

## Consequences

No Student-t DCC or ADCC in v1. Documented as Engle (2002) Gaussian DCC, not a full rmgarch clone.
