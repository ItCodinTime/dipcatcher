# ADR-033: quant-models engines (research only)

## Status

Accepted

## Date

2026-09-21

## Context

[davidalmeida90/quant-models](https://github.com/davidalmeida90/quant-models)
and the sibling repos named in that README are notebook engines for
BSM/Greeks, CRR, Heston, GBM, HRP, GEX, TSMOM, SVI, NSS, GKX, and
Krauss (2017). Dipcatcher already had GARCH/HAR, CVXPY MVO, and GKX
PCR/PLS/GBRT. It did not have option pricing, HRP, or the GEX last-hour
*rule*. The GEX bot places MES via IBKR; Dipcatcher does not.

## Decision

1. Port the **formulas** into `quant_fund.quant_models`, not the chart
   notebooks. `dipcatcher qm` is the CLI.
2. GEX last-hour is a **decision** (`last_hour_decide`). No IBKR, no
   Cboe fetch in the default path, `broker=None`.
3. Krauss is the **linear** sliding-window baseline. DNN/CNN/LSTM vol
   nets stay behind ADR-007 (`[nn]` extra). GKX R² vs zero is a metric
   on existing `pcr`/`pls`/`gbrt`.
4. These engines are not CS-ranker champions. `blend_weight` stays 0.
   Not a live P&L claim.

## Consequences

- Analytic greeks must pass `validate_greeks` (worst relative error
  `< 2e-4` vs finite differences of BSM).
- Heston uses the Albrecher little-trap CF.
- HRP is Lopez de Prado (2016) appendix code (single linkage + recursive
  bisection). Long-only min-variance stands in for CLA.
