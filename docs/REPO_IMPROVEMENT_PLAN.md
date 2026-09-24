# Repository improvement sequence

The goal is reviewable evidence of economic value and reliable operation.
Return improvement must be measured on the target market after costs.

| Order | Work | Status and acceptance evidence |
|---|---|---|
| 1 | Real-data benchmark | Runner, protocol, disclosures, matched baselines and 18 contract tests implemented. Lint and type checks pass. Real-data execution is pending recovery of the tracked Parquet snapshot; no market result is claimed. See REAL_DATA_BENCHMARK.md. |
| 2 | Net-return tournament | Implemented for equal-weight, momentum and reversal adapters. Frozen trial slate, causal common universe, shared next-open replay, costs/borrow/financing, two impact scenarios, full ledgers, validation-only selection and existing RC/SPA/StepM inference. The new bounded replay is independent of the legacy paths in PR #28. Real-market execution remains pending. See NET_RETURN_TOURNAMENT.md. |
| 3 | Cost-aware construction | Pending. Connect forecasts and uncertainty to feasible positions with turnover/impact/risk/capacity constraints. Require a matched ablation against the existing allocator after all costs. |
| 4 | Forward shadow record | Pending. Freeze the strategy before collecting subsequent outcomes; record decision times, orders, fills, rejects, positions and cash; reconcile restarts and benchmark results. Determine required evidence length from dependence and statistical power, not an arbitrary calendar count. |
| 5 | Performance and operations | Pending. Profile measured bottlenecks, establish representative workload budgets, preserve reference/fast-path economics, run required CI gates to completion, and test recovery and observability. |

The existing vendor pool discloses survivorship bias and already has results
for the 2025 holdout. It can support retrospective diagnostics. Stronger
evidence requires verified data availability/adjustments and a fresh externally
timestamped forward period. A passing forecast benchmark alone cannot promote
a strategy or authorize live execution.
