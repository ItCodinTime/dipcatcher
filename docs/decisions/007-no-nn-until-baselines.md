# ADR-007: No neural nets until baselines exist

## Status

Accepted (amended by ADR-023 for the robinhood+ / Kronos K-line engine only)

## Date

2026-09-15

## Context

Transformers and MDNs are easy to add and hard to validate. The spec forbids prioritizing them.

## Decision

Torch is an optional extra `[nn]`. Quantile nets, TCNs, LSTMs, and transformers are not implemented in v1. Tree and econometric baselines must pass leakage and cost-aware evaluation first.

## Consequences

PyTorch is not a default dependency. GPU is unused in CI.

## Amendment (2026-09-22): offline RL book-choice lane (dipcatcher.sota.v2 lane 4)

The dipcatcher.sota.v2 slate adds lane 4, a conservative offline contextual
bandit over the closed action set {cash, SPY, frozen top-5 12-1 momentum
book}. The reward is the next-bar net return after a 10 bp one-way switch
cost, not Sharpe. State features are public cross-sectional aggregates plus
trailing volatility and drawdown, all lagged one bar. The policy is trained
on the tape through 2024-12-31 only, frozen, and scored once on the
2025-01-02 holdout. It may replace the paper research overlay only if the
frozen holdout beats both SPY and the ungated top-5 inside the
pre-registered White reality check; it never moves blend_weight, submits
nothing, and is not a live P&L claim.
