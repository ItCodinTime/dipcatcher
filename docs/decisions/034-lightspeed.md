# ADR-034: Lightspeed engines (research only)

## Status

Accepted

## Date

2026-09-21

## Context

[cosmic-hydra/lightspeed](https://github.com/cosmic-hydra/lightspeed)
is Artificial Hedge's private book: frozen `tqqq-long-full-v1`
(QQQ 20/180 EMA → TQQQ/SGOV), `stock-momentum-v1` /
`nautica-momentum-v1` (top-1 63-day momentum + 200 SMA + crash
−20%/10d), and an AFML metalabel gate. The operator repo can
talk to Alpaca. Dipcatcher must not.

Yahoo expected-metrics in that repo ($655 / $283 paths, holdout
lost to SPY, metalabel PBO ~0.46) are **not** Dipcatcher live
claims.

## Decision

1. Port the **formulas** into `quant_fund.lightspeed`, not the
   Alpaca paper runner. CLI: `dipcatcher ls specs|demo`.
2. `live_disabled` is forced True. `broker=None`. No ALL-LIVE.
3. Metalabel can only **reduce** risk: multiplier ∈ [0, 1].
4. `nautica` is an a-priori CS challenger on `cs_z_mom_60`. It
   does not become champion. `blend_weight` stays 0. Public
   ridge remains the book until a non-SYNTHETIC public-feature
   card wins pairwise DM of −IC vs ridge plus White RC / SPA /
   StepM.

## Consequences

- Frozen params stay frozen: TQQQ EMA 20/180, flatten when
  fast < slow, `max_tqqq` 0.98, `vol_budget` 10, rebalance 5,
  delay 1; nautica `mom_fast` 63, `mom_blend` 1.0, `trend_sma`
  200, `top_k` 1, `vol_budget` 0.6, crash −0.2 / 10d, 10 bp.
- Lightspeed Yahoo dollar paths are not copied as Dipcatcher
  P&L. A file-tape race still has to clear the existing
  promotion gates.
