# ADR-022: Data-snooping suite (Reality Check / SPA / StepM / MCS) inside Dipcatcher

## Status

Accepted

## Date

2026-09-17

## Context

Dipcatcher already scores single-hypothesis inference (HAC t, Diebold–Mariano,
e-processes), FDR control (Benjamini–Hochberg), and backtest-overfitting
diagnostics (PBO/CSCV, DSR, PSR, MinTRL). What it lacked was the canonical
**data-snooping** layer: tests that ask whether the *best* strategy in a
universe of trials is real after accounting for every trial tried. This is the
question the lab exists to answer honestly, and the machinery (stationary
bootstrap, automatic block length, studentized recentering, stepwise FWER
control, model confidence sets) was missing.

Constraints: research-only (no live Sharpe / P&L claims), fail-closed on
malformed evidence, and no dependence on vendor data or live brokers.

## Options considered

### Option A: Rely on existing FDR + PBO only

- Pros: no new code; BH already handles a list of p-values.
- Cons: FDR needs per-strategy p-values (not a universe test); PBO needs CSCV
  splits of one strategy. Neither answers "is the max mean significant after
  snooping" under serial dependence.

### Option B: Circular block bootstrap only (existing helper)

- Pros: reuse `circular_block_indices`.
- Cons: the stationary bootstrap is the standard for the Reality Check / SPA /
  MCS family (strict stationarity of the resampled series), and a fixed block
  length is arbitrary — Politis–White automatic selection is part of the
  method's contract.

### Option C: Full suite — stationary bootstrap + Politis–White + RC + SPA
(lower/consistent/upper) + StepM + MCS

- Pros: canonical, citable, self-contained; integrates with the ranker universe
  already computed by the research agent; testable with size/power simulations.
- Cons: more surface to test; bootstrap cost on every research run (~1–2 s).

## Decision

Implement Option C.

- `metrics/inference.py`: `stationary_bootstrap_indices` (Politis–Romano) and
  `optimal_block_length` (Politis–White, flat-top kernel, documented clamps).
- `metrics/snooping.py`: `reality_check`, `spa_test` (lower / consistent /
  upper), `stepm` (step-down adjusted p-values), `model_confidence_set`
  (range statistic). Bootstrap p-values use `(1 + #{stat* ≥ stat}) / (B + 1)`.
- `validation/multiple_testing.py`: `TrialLedger.record_series` +
  `TrialLedger.snooping` battery over recorded trial series.
- Research agent: `families["ranking"]["data_snooping"]` over the aligned
  ranker date-IC series, minting **H99_ranking_data_snooping** (discovery) when
  the consistent SPA p-value is finite, with catalog honesty + verify wiring.
- Fail-closed rules: non-finite rows dropped with an explicit count;
  zero-variance columns dropped (studentization undefined); panels shorter than
  10 observations → honest NaN; shape violations raise `ValueError`; no
  forbidden research keys (no `sharpe`/`pnl`/`live_pnl_claim`).
- Docs: MATH_SPEC section, RESEARCH_REFERENCES entries (White 2000; Hansen
  2005; Romano–Wolf 2005; Hansen–Lunde–Nason 2011; Politis–Romano 1994;
  Politis–White 2004).

## Consequences

- The lab can answer "is the best ranker/trial real after snooping?" with a
  citable method, and StepM gives FWER-controlled per-strategy rejections.
- SPA's three p-values are reported for transparency; no universal ordering is
  asserted (the consistent variant equals the lower one exactly when no column
  is significantly negative, and can sit below it otherwise) — documented in
  MATH_SPEC and checked only for range/structure, not ordering.
- Hypothesis numbering: H46–H51 were claimed by the concurrent northset lane
  during development, so the ranking data-snooping hypothesis uses **H99** with
  headroom above that sequential lane.
- Research-only: the battery never claims live performance and is excluded from
  promotion gates.
