# Research 100

Dipcatcher has a [100-reference source-to-code catalog](RESEARCH100_CATALOG.md),
with 96 reused components and four new research components. This is **not 100
newly implemented or empirically reproduced papers**. References were selected
for relevance, complementary coverage, primary-source availability and existing
repository integration, rather than an unverifiable universal “top 100” ranking.

## What changed

- **R008 — Multi-period trading:** finite-horizon allocation with return forecasts,
  covariance, linear trading costs, quadratic impact, turnover limits and cash.
  Execute the first planned allocation and re-estimate at the next decision.
- **R009 — Risk-constrained Kelly:** discrete-scenario expected log growth with the
  Busseti–Ryu–Boyd negative-moment bound. The wealth floor is relative to initial
  wealth, not a trailing peak. The guarantee concerns the specified IID law;
  historical scenario estimation does not establish a market guarantee.
- **R010 — Volatility management:** inverse-variance exposures using strictly
  preceding returns, a training-only scale, variance floor and explicit cap.
  Rolling daily variance is a documented variant of the monthly paper.
- **R076 — Serial-adjusted Sharpe:** Lo's arithmetic time aggregation with a
  declared lag truncation, without assuming square-root annualization is valid.
- **CVaR correctness and memory:** report the optimized empirical tail objective,
  including fractional mass at the quantile. Sparse scenario auxiliaries replace
  the dense T-by-T identity, reducing constraint storage from O(T²+TN) to O(TN).
- **Risk parity correctness:** normalize the contribution objective so changing
  covariance units does not change the optimizer's stopping behavior. Reject
  indefinite covariance and unsuccessful portfolio solver results.
- **Evaluation:** causal walk-forward allocation, drifted weights, entry costs,
  an exact self-financing proportional-fee equation, and comparisons across all
  supplied trials using the existing White Reality Check and Hansen SPA.

These additions are opt-in research APIs and a CLI bench. They are not enabled
as live strategies and do not modify the existing promotion gates.

## Use

```bash
uv run dipcatcher research100 catalog
uv run dipcatcher research100 catalog R009
uv run dipcatcher research100 verify --output /tmp/research100-audit.json
uv run dipcatcher research100 benchmark --synthetic --output /tmp/research100-bench.json
uv run python scripts/render_research100.py
```

`verify` checks imports and hashes defining modules. It does **not** verify paper
equations or certify production readiness. The catalog JSON ships in the wheel.

```python
import numpy as np
from quant_fund.research.research100 import resolve_method
from quant_fund.metrics.research_evaluation import walk_forward_allocations

plan = resolve_method("R008")(
    forecasts=np.array([[0.001, 0.002], [0.0005, 0.001]]),
    covariance=np.diag([0.0001, 0.0002]),
    previous=np.zeros(2),
    risk_aversion=5.0,
    linear_cost=0.001,
)
first_decision = plan.weights[0]

# For your own aligned, point-in-time returns matrix:
# path = walk_forward_allocations(
#     returns, allocator_callback, lookback=252, fee_rate=0.001
# )
```

Methods retain their native signatures. Statistical tests, estimators and trading
policies are different objects; the registry does not treat all 100 as portfolio
strategies. Source-to-code bindings include approximation notes and linked tests.

## Evidence and limitations

The current CLI reports synthetic integration and accounting diagnostics only.
It omits Sharpe and total-return summary fields. The committed benchmark receipt
below is a historical snapshot from 2026-09-24; its module hashes and legacy
summary fields are preserved as recorded and do not describe the current code.

The committed [synthetic integration receipt](research100/synthetic_benchmark.json)
compares a fixed equal-weight baseline with inverse-volatility, multi-period,
risk-constrained Kelly and volatility-managed policies. It records all four
challengers, costs, observations, parameters, random seed, data hash, defining
module hashes, package versions and multiple-testing diagnostics. No search for
a favorable random seed or winning subset is performed. The dataset contains a
fixed volatility change and is explicitly labeled **SYNTHETIC**.

This short bench establishes integration and accounting behavior, not alpha.
Module hashes cover the listed modules, not every transitive dependency. Net
returns include proportional fees; there is no market-impact calibration, borrow,
financing, bid/ask replay or real fill model. Final holdings are marked rather
than liquidated. Annualized statistics from this short artificial sample are
illustrative only. The supplied fee rate must reflect the intended application.

For real data, the caller remains responsible for timestamp/calendar alignment,
corporate actions, delistings, as-of features, sample construction and all trials
attempted outside this comparison. Callbacks receive only copied historical
windows, but Python callbacks can still deliberately access external data.

Before claiming improved strategy performance, reproduce relevant equations and
original studies, pre-register economic objectives and baselines, use point-in-time
market data with realistic costs and capacity, retain every attempted candidate,
and evaluate an untouched final holdout plus paper execution. Improvements may
be absent or negative. No entry in this catalog is marked empirically reproduced
or supported by live-performance evidence.

## Validation for this change

[Validation record](research100/validation.json): 786 tests across 68 mapped
component suites passed, followed by 146 focused catalog/allocator/evaluation tests
on the final numerical implementation. These counts overlap and must not be added.
Ruff and type checks passed for the checked changes. Wheel and source-distribution
builds passed offline; the catalog CLI ran from the wheel in a separate dependency
environment. Alternate local virtual environments are now excluded from source builds.

The fixed synthetic comparison returned Reality Check p=0.85 and SPA p=1.0.
It does not support a mean net-return improvement claim. No real-market benchmark
or full-repository suite was rerun for this change. Existing component tests remain
component-level evidence, not complete equation audits of every linked paper.
