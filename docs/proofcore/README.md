# PROOFCORE

**PROOFCORE** is dipcatcher's proof-carrying, leakage-proof-by-construction,
statistically-honest backtest & research runtime. It turns the three audit
reports (correctness/leakage, architecture/performance, tests/CI) into one
coherent architecture layered *above* the existing packages — additive only,
no existing public API changes.

## What it adds

| Component | Package | What it does |
|---|---|---|
| Shared contracts | `quant_fund.proofcore.contracts` | pydantic schemas (`ProofBundleV1`, `TrialLedgerRow`, `LeakageReport`, `RealityReport`), canonical hashing, error taxonomy |
| PIT Vault (W1) | `quant_fund.pit` | Write-once, content-addressed, bitemporal store; `asof(t)` is the ONLY legal read path |
| Proof-Carrying Backtester (W2) | `quant_fund.proof` | Every run emits a signed, hash-chained `ProofBundleV1`; the verifier re-derives every hash AND recomputes headline metrics from the trade log |
| Leakage Hunter (W3) | `quant_fund.leakage` | AST linter (LH001–LH012), runtime watchdog, seeded-leak fixture suite |
| Reality Filter (W4) | `quant_fund.reality` | Unit-safe PSR/MinTRL, DSR with effective trials, CSCV/PBO, SPA, BH-FDR over the trial ledger |
| Provenance & CI (W5) | `quant_fund.proofcore.provenance`, `.github/workflows/proofcore.yml` | duckdb provenance DB, receipts re-verification, per-package coverage floors, layering gate |

## Quickstart

```bash
# provenance ledger (duckdb at data/metadata/proofcore.duckdb — gitignored)
quant proofcore log --bundle proofs/bundles/<id>.json
quant proofcore query
quant proofcore export --out data/metadata/proofcore-trials.jsonl
quant proofcore chain-head

# proven run + verification (W2)
quant proof run --config configs/research.yaml --seed 7 --pit-root data/pit --bundle-dir proofs
quant proof verify --bundle proofs/bundles/<id>.json --replay

# leakage scan (warn mode this wave)
quant leakage scan --paths src/quant_fund --format json

# reality filter over the exported trial ledger (W4)
quant reality trial-report --ledger data/metadata/proofcore-trials.jsonl --out report.json
quant reality ledger-gate --ledger data/metadata/proofcore-trials.jsonl
```

## Local gates

```bash
make proofcore-test       # contracts, provenance, CI-helper, layering tests
make proofcore-coverage   # per-package floors: pit/proof/reality/proofcore 90, leakage 85
make proof-verify         # double-run bundle identity + tamper evasion
make leakage-scan         # warn mode this wave (adjudicated)
make reality-gate         # trial-report + ledger-gate (advisory in CI at first)
make receipts-reverify    # re-verify every committed receipt (A3 #4)
```

## Honesty contract

PROOFCORE strengthens, never weakens, the AGENTS.md honesty rules. No
PROOFCORE CLI prints Sharpe/P&L/NAV as a headline — reality reports are
diagnostics, and bundle `metrics_recompute` exists only as a verification
artifact inside signed proofs.

See `MIGRATION.md` for the additive rollout phases.
