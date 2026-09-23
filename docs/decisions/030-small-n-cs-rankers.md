# ADR-030: Small-N CS rankers and v4 public features

## Status

Accepted

## Date

2026-09-20

## Context

Wave 148’s 55-name 5-day v3 race showed public ridge IC −0.011 after the
MAD fallback. OLS Fama–MacBeth needs \(N>p+2\); 54 names and 39 columns
make per-date OLS unidentified. sklearn pooled `Ridge(α=1)` is OLS on a
100k-row tape. The wide tape’s FM t-stat (2.61) is the same estimator
with enough names.

## Decision

1. When dates are passed, champion ridge date-demeans \(y\) and uses
   \(\alpha T\). Tests that omit dates keep the old \(\alpha\).
2. Catalog `fm_ridge`, `classic`, and `combo_ic`. Signs and IC weights
   are a priori / train-fold only. `blend_weight` stays 0.
3. `features.v4`: skip-momentum, residual momentum, rank-space reversal
   and overnight on the public card.

## Consequences

- File-tape date IC vs public ridge remains the promotion gate.
- 55-name and wide 5-day races must rebuild gold (`features.v4`).
- Not a live P&L claim. SYNTHETIC IC is still correctness, not promotion.
