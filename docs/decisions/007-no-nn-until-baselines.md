# ADR-007: No neural nets until baselines exist

## Status

Accepted

## Date

2026-09-15

## Context

Transformers and MDNs are easy to add and hard to validate. The spec forbids prioritizing them.

## Decision

Torch is an optional extra `[nn]`. Quantile nets, TCNs, LSTMs, and transformers are not implemented in v1. Tree and econometric baselines must pass leakage and cost-aware evaluation first.

## Consequences

PyTorch is not a default dependency. GPU is unused in CI.
