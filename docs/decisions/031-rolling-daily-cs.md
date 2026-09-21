# ADR-031: Rolling daily-CS window and short-horizon rankers

## Status

Accepted

## Date

2026-09-21

## Context

Wave 149’s 55-name 5-day v4 race left public ridge at IC ≈ 0. Mixed
`classic` (monthly Amihud / 52-week / 12-1 plus daily reversal) was
~0. `combo_ic` led at 0.039 (t=1.93) but pairwise DM vs ridge did not
reject at 5%. Walk-forward was expanding, so late folds pooled ~10y of
54 mega-caps. Daily reversal is short-memory (Jegadeesh 1990; Lehmann
1990). `hedge_lab.yaml` already set `train_bars: 252` but inherited
`scheme: expanding` from `sota_file.yaml`.

## Decision

1. `configs/hedge_lab.yaml` uses `validation.scheme: rolling` with a
   252-session train window. The wide tape inherits it.
2. Catalog daily-CS challengers whose column masks and signs are a
   priori: `reversal`, `classic_st`, `ridge_st`, `fm_st`, `combo_ic_st`,
   and Rapach discounted nested-MSFE `combo_msfe`.
3. OOS score cache keys include scheme / train / val / test / embargo
   so expanding artifacts cannot be reused as rolling scores.
4. Champion stays public ridge. `blend_weight` stays 0. Promotion still
   requires pairwise Diebold–Mariano of −IC vs ridge plus White RC /
   SPA / StepM on a non-SYNTHETIC public-feature card.

## Consequences

- 55-name live `ranking_target` remains `future_idio_return_1` (from
  `sota_file.yaml`). Paper races run 1-day and 5-day labels separately.
- Not a live P&L claim.
