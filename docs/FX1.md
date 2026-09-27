# fx-1

**fx-1** (always lowercase, package `src/fx1`) is an in-tree sub-project. It
contains the corpus builder, the eval bank, and training plumbing, plus a
training plan. No trained checkpoint is in this repository. Local generation
is unimplemented. The declared base constant is `moonshotai/Kimi-K3` (2.8T
total / 104B active MoE, Kimi K3 License).

**dipcatcher** is the harness: the data engine, evaluation bench, and
verification layer. `src/fx1/harness.py` is the typed bridge; `fx1 harness list`
shows the registered lab surfaces.

## What the plumbing enforces

The table is the corpus and eval contract. It is a plan for a future training
run, and a record of what the package checks today.

| Property | General LLM | fx-1 plan |
|---|---|---|
| Training data | Web-scale text | Only dipcatcher artifacts whose receipts pass `verify-research` — notebooks, scorecards, bench outputs, tournament ledgers, gate decisions |
| Negative examples | None | Gate rejections and live-claim artifacts, teaching refusal and honest reporting |
| Provenance | None | Every training example carries the SHA-256 of its source receipt |
| Honesty contract | Prompt scaffolding | Tested natively: `fx1.honesty` + eval harness block forbidden Sharpe/P&L headlines, live-performance claims, unlabeled synthetic evidence |
| Evidence classes | Blurred | Explicit: research / backtest / simulated paper / SYNTHETIC |

## Package layout (`src/fx1/`)

- `data/receipts.py` — receipt loading and eligibility (`research_only=true`,
  `live_pnl_claim=false`); ineligible artifacts become negative examples.
- `data/corpus.py` — SFT corpus builder → JSONL, one `SFTExample` per line,
  each with `receipt_sha256` provenance. Run:
  `python -c "from fx1.data import build_corpus; build_corpus('receipts', 'data/fx1/corpus.jsonl')"`
- `eval/suite.py` — deterministic eval harness (honesty / domain / general
  task kinds). Runs **before** any training; the honesty gate blocks shipping.
- `train/config.py` — validated run config: ladder stage (`proxy` →
  `final_k3` → `distill`), LoRA hyperparameters, mandatory cost disclosure,
  K3 constraints (multi-node, preserved `reasoning_content`).
- `train/run.py` — `build_training_manifest`: enforces eval-before-train and
  corpus provenance, then writes an immutable run manifest
  (`live_pnl_claim=false`, `research_only=true`) for the cluster launcher.
- `prompts/fx1_system.md` — the system prompt / behavioral contract written
  into the corpus builder.

## License

Kimi K3 License: internal/research use is unrestricted. A Model-as-a-Service
business above $20M revenue requires a separate Moonshot agreement; above
100M MAU or $20M monthly revenue requires UI attribution. Current tier:
**internal_research**. Re-check at every fx-1 release.

## Hard boundaries (inherited from the lab)

The package leaves lab data, gates, receipts, and promotion logic to the
harness. It refuses live-performance claims and keeps synthetic results
labeled as synthetic. These rules are enforced in code (`tests/fx1/`) and in
the corpus builder.
