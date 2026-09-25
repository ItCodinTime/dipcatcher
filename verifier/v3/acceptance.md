# Acceptance criteria v3 - industry grade

Extends v2 (all v1/v2 criteria still hold). New:

1. Data quality gates: exact dedup, near-dup shingle screen, eval
   contamination removal (fail-closed), length stats, frozen seeded split
   with hash manifest.
2. Tool-use traces: reasoning_content + tool_calls preserved; verification-
   gated admission; failed trajectories marked negative.
3. DPO preference pairs per honesty-bait class with disjoint chosen/rejected.
4. Immutable training receipts: git revision + dirty flag + config/corpus/
   split/eval hashes + seed + env fingerprint; tamper-evident verification.
5. Statistical ship gate: bootstrap CI + McNemar on paired pass outcomes.
6. Cluster specs: validated (multi-node, ZeRO, precision, long-context rules)
   and generated artifacts.
7. Staged pipeline: data->quality->eval_base->train->eval_candidate->card
   with gate-order enforcement and honesty-gated base eval.
8. CI workflow for fx1 gates; experiment tracking wrapper (MLflow/JSONL).
9. Quality: full tests/fx1 green; ruff clean.
