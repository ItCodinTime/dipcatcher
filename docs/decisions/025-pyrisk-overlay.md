# ADR-025: pyRisk / pyriskmgmt paper-book overlay

## Status

Accepted

## Date

2026-09-19

## Context

The first hedge-lab book printed Sharpe −1.05 and max DD −25% because the
backtest flattened to cash on every day without a weight row. The user asked
to implement [pyRisk](https://github.com/lprtk/pyRisk) and
[pyriskmgmt](https://github.com/GianMarcoOddo/pyriskmgmt) and to push Sharpe /
Calmar / Sortino / returns up while holding max DD under 5%.

Risk engines measure and *cap* loss. They do not mint alpha. A 5% drawdown
budget on a 16% vol book is a vol target near 2.5%, not an exponential Sharpe
machine. Hold-until-rebalance is a correctness fix, not a new signal.

## Decision

1. Implement the pyRisk contract in `quant_fund.risk.pyrisk` (empirical /
   Gaussian / Monte Carlo / Hill-EVT VaR-ES, Pickands, Leadbetter, Kupiec /
   Christoffersen) and the pyriskmgmt equity subset in
   `quant_fund.risk.pyriskmgmt` (EWMA VaR-ES, portfolio aggregation, component
   VaR, scale-to-ES). MIT attribution in module docs. No pip dependency on
   those repos (pyRisk wants PyWavelets; pyriskmgmt's derivative module is
   ~470 KB of option engines we do not trade).
2. `BookRiskOverlay` scales `run_backtest` target weights using **prior close
   NAV only**: default vol target `dd_limit/2` (2.5% for a 5% DD budget), EWMA
   / historical ES cap, a 20% vol prior until the lookback fills, and a
   remaining-drawdown halt with a 50 bp cushion. One-day gaps can still
   overshoot slightly.
3. Overlay diagnostics live in `hedge_lab_analytics`, not research family
   blobs. `blend_weight` stays 0. `live_pnl_claim` stays false.
4. Sparse rebalance panels **hold** the last target until the next decision
   date. Missing weight rows are not a flatten-to-cash instruction. The
   overlay can still scale or flatten those carried weights from prior-close
   NAV.

## Consequences

- Max DD can be pulled toward 5% on the same causal ridge book by targeting
  vol at `dd_limit/2` rather than parking in cash after the first 5% cut.
- `dipcatcher hedge-lab --no-risk-overlay` is the hold-until-rebalance book
  (not the old daily flatten). Risk engines do not mint alpha; they cap loss.
