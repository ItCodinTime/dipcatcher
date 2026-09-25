# fx-1 Architecture — the map an auditor reads first

## The inversion in one paragraph

fx-1 is the product: a quant LLM fine-tuned from Kimi K3 open weights.
dipcatcher is the harness: the data engine, evaluation bench, and
verification layer that builds and tests fx-1. The repo is organized so the
model's *integrity is inherited from the harness's gates* — and then made
externally verifiable through four uniqueness moves (masked eval, attested
inference, corpus ledger, MRM dossier).

## Layer diagram

```
                         ┌─────────────────────────────┐
                         │  SERVING  (fx1.serve)        │
                         │  hosted K3 · local fx-1      │
                         │  release signing · TEE/zkML  │
                         │  cited_complete (honesty+    │
                         │  provenance at inference)    │
                         └──────────────┬───────────────┘
                                        │ model cards + ship gate
┌──────────────────────────────────────────────────────────────────┐
│ TRAINING  (fx1.train)                                            │
│  staged pipeline: data→quality→eval_base→train→eval_cand→card    │
│  LoRA ladder · DPO pairs · cluster specs · immutable receipts    │
│  curriculum · experiment tracking                                │
└──────────────┬───────────────────────────────────┬───────────────┘
               │ corpus                            │ scores
┌──────────────▼───────────────┐   ┌───────────────▼───────────────┐
│ DATA  (fx1.data)             │   │ EVAL  (fx1.eval)              │
│  receipts·notebooks·ledgers· │   │  task bank · red team ·       │
│  traces → SFT corpus         │   │  masked twins + memory gap ·  │
│  quality gates · frozen      │   │  time partitions ·            │
│  splits · HASH-CHAINED LEDGER│   │  contamination audit · stats  │
└──────────────┬───────────────┘   └───────────────┬───────────────┘
               │                                   │
        ┌──────▼───────────────────────────────────▼──────┐
        │ HARNESS  (fx1.harness → dipcatcher / quant_fund) │
        │  23 registered lab commands, 4 roles, fail-closed│
        │  benches (proper scores) · verify-research ·     │
        │  honesty gates (FORBIDDEN metrics) · PIT data    │
        └──────────────────────────────────────────────────┘
```

## Trust flow (what an auditor should trace)

1. **Corpus example** → carries `receipt_sha256` of a harness artifact that
   passed `verify-research`; recorded in the hash-chained ledger.
2. **Training run** → blocked until quality gates, contamination scan, frozen
   split, and a passing base eval exist; emits an immutable receipt.
3. **Checkpoint** → ships only with a model card whose eval deltas beat the
   base statistically (bootstrap CI + McNemar) and whose honesty gate passed
   natively; signed release; optional TEE/zkML attestation.
4. **Served answer** → honesty-validated at inference, provenance-footered;
   refusal behaviors trained via DPO and red-teamed adversarially.
5. **Regulatory dossier** → `fx1 mrm` compiles all of the above into the
   five-activity structure.

## Invariants (all enforced in tests)

| Invariant | Enforcement |
|---|---|
| No forbidden headline metrics as research results | `fx1.honesty` + lab catalog gates + DPO + red team |
| No live-performance claims | receipt eligibility + ledger scan + card validator |
| No unsigned serving (when keyed) | `LocalFx1Backend` signature check |
| No training without eval + receipt | pipeline stage gates |
| No corpus without provenance | corpus builder + quality gates + ledger |
| No eval leakage | contamination audit + masked twins + time partitions |

## Where the boundaries honestly are

- GPU trainer binding is an injectable seam (needs a cluster).
- TEE crypto verification delegates to platform SDKs at deployment.
- zkML tier is operator-manifest scaffolding pending prover maturity.
- Post-cutoff eval grows as new harness artifacts accumulate.
