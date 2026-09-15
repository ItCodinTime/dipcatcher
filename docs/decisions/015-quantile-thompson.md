# ADR-015: Quantile Thompson sampling as a LinUCB sibling

## Status

Accepted

## Date

2026-09-15

## Context

LinUCB ranks a date's names by a shared linear mean plus an exploration bonus. That is a point-forecast bandit. Residual reward is heteroskedastic and skewed, so a sampled quantile of the conditional law can rank differently from the mean. Neural distributional RL (IQN, QR-DQN) is blocked by ADR-007. The lab score remains scientific residual reward versus a uniform policy, not a portfolio path metric.

## Options Considered

### Option A: Keep LinUCB only
- Pros: already implemented; closed-form ridge updates.
- Cons: mean-only; no posterior draw over a reward law.

### Option B: Neural quantile / distributional RL
- Pros: flexible conditional quantiles.
- Cons: forbidden until baselines exist (ADR-007); not default-installable.

### Option C: Shared linear quantile Thompson (IRLS pinball + posterior draw)
- Pros: numpy-only; deterministic given `seed`; distributional; same panel/trace contract as LinUCB.
- Cons: IRLS is an approximation to pinball; Gaussian posterior is a Laplace stand-in.

### Option D: Mean Bayesian ridge plus quantile residual bootstrap only
- Pros: cheaper; still samples a tau-quantile of predictive reward.
- Cons: a homoskedastic residual quantile is a constant and does not change ranks without a scale model.

## Decision

We choose **Option C**. `QuantileThompson` keeps one shared linear pinball model per tau on the grid `j / (m + 1)`, `j = 1..m` (default `m = 9`). Each update appends `(x, r)` and refreshes Bayesian ridge sufficient statistics. On the next `select`, IRLS refits every tau, warm-started from the ridge mean, and stores the weighted precision `X'WX + λI`. Selection draws one tau uniformly, draws `β_τ ~ N(β̂_τ, (X'WX + λI)^{-1})`, and returns the top-k rows of `X β_τ`. `run_panel` walks date groups in time order, scores that date only, then updates the chosen arms — dates are never stacked for a decision.

## Consequences

- New module `quant_fund.models.quantile_bandit` is standalone; `rl.py` is unchanged.
- Promotion evidence is mean policy residual versus uniform and cumulative regret versus an in-date oracle.
- Crossing of the raw tau-grid is not repaired; a single sampled tau is used for ranking.
