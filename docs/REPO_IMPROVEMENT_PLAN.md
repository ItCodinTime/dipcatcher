# Repository improvement sequence

The goal is reviewable evidence of economic value and reliable operation.
Return improvement must be measured on the target market after costs.

| Order | Work | Status and acceptance evidence |
|---|---|---|
| 1 | Real-data benchmark | Executed against the tracked 424-name snapshot with sealed validation and previously inspected test scores. Zero forecast won validation MSE; ridge's small test advantage has no economic interpretation. See REAL_DATA_BENCHMARK.md and the Phase 1 evidence report. |
| 2 | Net-return tournament | Executed the frozen equal-weight, momentum-20 and reversal-1 slate with full ledgers and both impact scenarios. Validation selected momentum-20, which underperformed equal weight on the inspected test; the economic evidence gate is false. See NET_RETURN_TOURNAMENT.md and the Phase 1 evidence report. |
| 3 | Cost-aware construction | The matched validation run retained both failed cost-aware candidates: CLARABEL returned `optimal_inaccurate`. No candidate was selected, so there is no cost-aware test or uplift estimate. Constraints remain fail-closed. See COST_AWARE_CONSTRUCTION.md and the Phase 1 evidence report. |
| 4 | Forward shadow record | The bounded paired simulated paper adapter and pre-collection 1,400-session power plan are implemented; no external freeze or prospective decision has been collected. An eligible timestamped complete-universe feed and independently recorded freeze are required before collection. See PAPER_SHADOW.md and FORWARD_SHADOW_POWER.md. |
| 5 | Performance and operations | Pending. Profile measured bottlenecks, establish representative workload budgets, preserve reference/fast-path economics, run required CI gates to completion, and test recovery and observability. |

The existing vendor pool discloses survivorship bias and already has results
for the 2025 holdout. It can support retrospective diagnostics. Stronger
evidence requires verified data availability/adjustments and a fresh externally
timestamped forward period. A passing forecast benchmark alone cannot promote
a strategy or authorize live execution.

The sealed Phase 1 report is `data/metadata/research/phase1_20260925.md`;
`data/metadata/research/phase1_evidence_index.json` links its complete and
blocked runs to the frozen configs and source commit. Use
`dipcatcher verify-research data/metadata/research/phase1_evidence_index.json`
in a checkout containing the tracked snapshot and published ledgers.
