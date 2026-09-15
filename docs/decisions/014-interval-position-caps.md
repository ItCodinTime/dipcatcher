# ADR-014: Conformal intervals become executable name caps

## Status

Accepted

## Date

2026-09-15

## Context

Forecasts already carry conformal sets (`interval_lo`, `interval_hi`). Those sets stop at coverage and width (ADR-008, ADR-009). The book still uses a single `name_max` / risk-gate cap. A wide set, or a lower endpoint deep in the left tail, is not a reason to keep the full name weight. This module is the missing map from sets to a per-name position cap. It does not invent P&L or Sharpe.

## Options considered

### Option A: Inverse-width × inverse-downside product (chosen)

\[
\mathrm{cap}_i=\bar w\cdot\frac{w_{\mathrm{ref}}}{w_{\mathrm{ref}}+\mathrm{width}_i}\cdot\frac{d_{\mathrm{ref}}}{d_{\mathrm{ref}}+\max(0,-\ell_i)}
\]

- Pros: strictly monotone in width and downside; caps stay in \([0,\bar w]\); point interval at 0 keeps \(\bar w\); deterministic.
- Cons: never hard-zeros a finite valid set (only missing / inverted / non-finite sets hit 0).

### Option B: Linear remaining budget \(\max(0,1-\mathrm{width}/w_{\mathrm{ref}})\)

- Pros: hits a hard floor once width exceeds the reference.
- Cons: not strictly monotone after the floor; two very different wide sets get the same zero cap.

### Option C: Binary “zero in the set ⇒ weight 0”

- Pros: simple sign-uncertainty rule.
- Cons: ignores width; a tight interval around a slightly negative \(\ell\) is treated like a huge set.

### Option D: Clip then silently renormalize long-only to the original gross

- Pros: keeps a full book.
- Cons: hides de-risk; a single wide name inflates the others. Forbidden.

## Decision

Use **Option A** in `quant_fund.portfolio.interval_risk`. `apply_interval_caps` returns `(capped_weights, caps)` and does **not** renormalize. Missing, NaN, infinite, or inverted intervals fail closed to cap 0. `bench_interval_caps` reports mean cap, fraction binding, and mean width only — no Sharpe.

`risk_gate.py` and `optimizer.py` stay unchanged. Callers pass these caps as an extra box or a post-optimizer clip.

## Consequences

A name with a tight, non-negative lower set can still reach `max_weight`. A name whose conformal interval is wide or whose lower tail is deeply negative is size-capped before it reaches the gate. Gross after the clip can be smaller than gross before; that shrinkage is the feature.
