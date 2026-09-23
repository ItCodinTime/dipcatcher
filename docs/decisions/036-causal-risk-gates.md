# ADR-036: Causal risk-controlled gates (research only)

## Status

Accepted

## Date

2026-09-21

## Context

The 55-name public CS tape has near-zero IC. Joint targets Sharpe > 5,
max DD < 5%, and 10×/50×/100× net P&L are overconstrained at a 2.5%
vol target (Sharpe 5 is only ~12.5%/year ≈ 3.4×, not 10×). ADR-025
already scaled the paper book with vol/DD/ES. The Wave 156 hunt added
Kelly, CRC, crash, and expanding-window StepM as delay-1 size gates.
They cap loss. They cannot mint Sharpe 5 from IC ≈ 0 (constant
leverage leaves Sharpe unchanged when rf = 0; ADR-032 forbids
sign-flipping a loser).

## Decision

1. Implement causal gates in `quant_fund.risk.gates` and take `min`
   with `BookRiskOverlay` Kelly/CRC (lookback 63). Delay 1.
2. `tsmom` / `vme` / `krauss` are public-feature CS challengers, not
   champions. Promotion still needs pairwise DM of −IC vs ridge plus
   White RC / SPA / StepM on a non-SYNTHETIC card.
3. `blend_weight` stays 0. Not a live P&L claim.
4. Holdout confirmation of a pre-declared challenger uses the frozen
   Lightspeed cut (`SELECTION_END=2024-12-31`,
   `HOLDOUT_START=2025-01-02`). Engine choice that already saw the
   full OOS IC is a persistence check, not a retune.

## Consequences

- Wave 156: `tsmom` won DM (p=0.014), SPA (p=0.028), StepM; White RC
  p=0.057 missed 5%. **Not promoted.**
- Wave 157 holdout (n=189 from 2025-01-02): tsmom IC +0.011 t 0.54;
  DM p=0.60; RC p=0.44; SPA p=0.41; StepM []. **Did not persist.**
  Selection-only (n=945) would have cleared all four gates; that is
  not a promotion after seeing holdout.
- CLI: `dipcatcher ls hunt|book|race|confirm`.
- Do not invert `classic_st`. Do not add 12–1 lookalikes after seeing
  holdout.
