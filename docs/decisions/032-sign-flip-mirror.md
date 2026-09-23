# ADR-032: Sign-flip mirror books

## Status

Accepted

## Date

2026-09-21

## Context

A natural hope is: find a strategy with Sharpe \(< -5\) and \(-150\%\)
P&L, flip the signs, and harvest Sharpe \(> 5\) with a 5% drawdown. Three
identities block that as a free lunch.

1. Sharpe is odd and leverage-invariant (rf = 0, no costs): \(\mathrm{SR}(c r)=\mathrm{sign}(c)\,\mathrm{SR}(r)\). Levering a mildly negative book to \(-150\%\) does not create Sharpe \(-5\).
2. Spread, commission, and impact are even. Both the book and the mirror pay them. Random high-turnover scores lose both ways.
3. A 5% peak-to-trough halt (`BookRiskOverlay.dd_limit=0.05`) cannot coexist with \(-150\%\) total return on the same path.

Searching for the worst in-sample univariate and flipping it is the same search as picking the best in-sample univariate.

## Decision

1. Catalog `negate_target_weights` and date-level dollar-neutral long-short of public walk-forward scores vs their sign-flip. `dipcatcher hedge-lab --mirror` backtests flipped `target_weight` on the same fills and cost model.
2. Nested `anti_univ` selects the *lowest* train-fold date IC, then scores that column OOS. Its mirror is best-train-IC, not a new edge.
3. Receipts stay `paper_backtest`. `blend_weight` stays 0. `flag_high_sharpe` still fires at \(|S|>5\). SYNTHETIC oracle Sharpe is not promotion.

## Consequences

- File-tape mirror numbers are diagnostics, not a live P&L claim.
- Champion remains public ridge until pairwise DM of −IC plus White RC / SPA / StepM.
