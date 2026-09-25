# fx-1 Data Contract — industry-grade corpus pipeline

Every byte fx-1 trains on passes through this pipeline. No exceptions.

## Sources (all provenance-hashed)

| Source | Module | Content | Gate |
|---|---|---|---|
| Harness receipts | `fx1.data.receipts` | Verified lab results | `research_only=true`, `live_pnl_claim=false` else negative example |
| Research notebooks/docs | `fx1.data.notebooks` | Contracts, reasoning, evidence classes | chunked on sections, source SHA-256 |
| Ledger artifacts | `fx1.data.ledgers` | Tournament/backtest evidence | recursive `live_pnl_claim` scan → negative |
| Tool-use trajectories | `fx1.data.traces` | K3 teacher sessions | `verify_ok=true` for positive; reasoning_content + tool_calls preserved (K3 requirement) |

## Quality gates (`fx1.data.quality`)

1. **Exact dedup** — SHA-256 of normalized message text.
2. **Near-dup screen** — 8-token shingle Jaccard ≥ 0.9 against the recent window.
3. **Eval contamination** — any example with ≥0.6 shingle containment against the eval bank prompts is removed fail-closed. Leaked evals are how labs lie to themselves.
4. **Length policy** — 32k-char cap; p50/p99/max reported in `QualityReport`.
5. **Frozen split** — deterministic seeded train/val split with `SplitManifest` (hashes + counts). Training receipts cite the split manifest hash, so any run can prove exactly which examples it saw.

## Preference data (`fx1.train.dpo`)

DPO pairs are generated per honesty-bait class: shared prompt, chosen =
contract-honoring refusal/report, rejected = the exact forbidden behavior
(Sharpe headline, live P&L claim, synthetic-as-live, gate relaxation,
guaranteed returns). Preference training is what makes honesty *native* rather
than prompt-scaffolded.

## Format

One JSON object per line (`SFTExample`): `messages` (system/user/assistant,
with `<reasoning>` and `<tool_call>` blocks for trajectories),
`receipt_sha256` (provenance), `source_path`, `negative` flag.
