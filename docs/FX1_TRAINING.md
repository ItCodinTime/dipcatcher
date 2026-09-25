# fx-1 Training Plan — compute, ladder, and gates

## Hardware reality (K3 base)

| Fact | Consequence for fx-1 |
|---|---|
| K3 checkpoint ≈ 1.4TB MXFP4 (≈5.6TB dequantized BF16) | No local training; all runs on rental clusters |
| Serving needs ≈64 accelerators; smallest community 1-bit quant ≈610GB RAM+VRAM | Production fx-1 is a **distilled student**, not raw K3-LoRA |
| 104B active parameters per token | Even LoRA touches multi-node memory; FINAL_K3 runs need ≥2 nodes (enforced in `TrainConfig`) |
| Kimi Delta Attention + native MXFP4 QAT | Use only frameworks with confirmed K3 support (check vLLM/SGLang/Unsloth status at run time); keep QAT formats intact |
| Preserved-thinking-history training | Corpus must keep `reasoning_content` + `tool_calls` (enforced in `TrainConfig`) |

## The ladder

1. **PROXY** — iterate corpus, eval harness, and LoRA plumbing on a small
   open model or quantized K3 serving build. Cost: hundreds of USD per run.
2. **FINAL_K3** — LoRA/QLoRA on the full K3 base via a rental multi-node
   cluster. Cost estimate committed to this file *before* launch; a serious
   run is five to six figures USD. v0.x never full-fine-tunes.
3. **DISTILL** — distill the K3-LoRA teacher into a smaller servable fx-1
   student so production inference fits a single node.

## Pipeline contract (enforced in code)

1. Build corpus: `fx1.data.build_corpus` — every line carries
   `receipt_sha256`; ineligible receipts become negative examples.
2. Run eval harness on the **base model** and record results — training is
   blocked without an on-record eval summary whose honesty gate passed.
3. `build_training_manifest` validates corpus provenance + eval gate + cost
   disclosure and writes the immutable run manifest.
4. Train on cluster; log checkpoints as `fx-1.vX.Y` with a model card:
   base checkpoint hash, corpus receipt range, domain/general eval deltas,
   license tier.
5. Ship gate: fx-1 beats the K3 base on domain tasks, does not regress on
   general tasks, and passes all honesty tasks natively.

## Corpus status

Seed corpus built from `receipts/`: 5 receipts loaded, 5 eligible (positive),
0 ineligible. Since v8 the loader also recognizes the lab's research-run
manifest schema (`data/metadata/research/runs/*.json`: `claim: "research_only"`
+ `synthetic` flag): 88 manifests → 77 positive (all labeled SYNTHETIC —
simulated-data evidence, never market evidence) + 11 negative (no claim →
fail-closed refusal examples). The corpus grows automatically as Phases 1–3 of
the roadmap (real-data benchmark, tournament, Dip Quality Score bench,
leaderboard) produce new gate-passed receipts — the lab's research output *is*
fx-1's training data flywheel.

## Industry-grade pipeline (v3)

Training runs follow a six-stage gated pipeline (`fx1.train.pipeline`):
`data → quality → eval_base → train → eval_candidate → card`. A stage that
cannot produce its evidence stops the pipeline — no skip flags.

- **Quality gate:** dedup, near-dup shingle screen, eval-contamination
  removal, frozen split manifest (`fx1.data.quality`).
- **Eval-before-train:** base-model results on the eval bank are mandatory
  and hashed into the run.
- **Immutable training receipts** (`fx1.train.receipts`): git revision,
  dirty-worktree flag, config/corpus/split/eval hashes, seed, environment
  fingerprint; `verify_training_receipt` re-checks hashes against disk.
- **Statistical ship gate** (`fx1.eval.compare`): paired bootstrap CI on
  pass-rate deltas + McNemar; domain improvement must exclude zero.
- **Cluster configs generated, not hand-edited** (`fx1.train.cluster`):
  validated multi-node ZeRO-3 specs for the 2.8T MoE; long-context runs
  require ≥8 nodes; MXFP4 QAT runs require ZeRO-3.
- **Experiment tracking** (`fx1.train.tracking`): MLflow when available,
  JSONL always; tracking is observability — the receipt is the evidence.
- **Preference stage** (`fx1.train.dpo`): contract-derived DPO pairs make
  honesty native.
- **CI** (`.github/workflows/fx1.yml`): fx1 tests, lint, blocking honesty
  inheritance, and a corpus-contract smoke run on every fx1 change.
