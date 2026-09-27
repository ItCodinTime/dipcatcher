# PROOFCORE

**PROOFCORE** is an experimental set of PIT storage, proof foundations,
leakage checks, and statistical diagnostics layered above the existing
research packages. Bundle replay verification and proven-run orchestration
remain pending.

## What it adds

| Component | Package | What it does |
|---|---|---|
| Shared contracts | `quant_fund.proofcore.contracts` | pydantic schemas (`ProofBundleV1`, `TrialLedgerRow`, `LeakageReport`, `RealityReport`), canonical hashing, error taxonomy |
| PIT Vault (W1) | `quant_fund.pit` | Write-once, content-addressed, bitemporal store; `asof(t)` is the ONLY legal read path |
| Proof foundations (W2) | `quant_fund.proof` | Environment fingerprinting, data-read recording, and signer tests; the runner and replay verifier are pending |
| Leakage Hunter (W3) | `quant_fund.leakage` | AST linter (LH001–LH012), runtime watchdog, seeded-leak fixture suite |
| Reality Filter (W4) | `quant_fund.reality` | Unit-safe PSR/MinTRL, DSR with effective trials, CSCV/PBO, SPA, BH-FDR over the trial ledger |
| Provenance & CI (W5) | `quant_fund.proofcore.provenance`, `.github/workflows/proofcore.yml` | duckdb provenance DB, receipts re-verification, per-package coverage floors, layering gate |

## Quickstart

```bash
# provenance ledger (duckdb at data/metadata/proofcore.duckdb — gitignored)
quant proofcore log --bundle path/to/existing-bundle.json  # logged as unverified
quant proofcore query
quant proofcore export --out data/metadata/proofcore-trials.jsonl
quant proofcore chain-head

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
make proof-integrity      # current signer/recorder tests
make leakage-scan         # warn mode this wave (adjudicated)
make reality-gate         # trial-report + ledger-gate (advisory in CI at first)
make receipts-reverify    # fail-closed audit; heterogeneous receipt verifiers pending
```

## Honesty contract

PROOFCORE follows the AGENTS.md honesty rules. Logged bundles remain
unverified until a replay verifier is implemented; `--verification` is
rejected. Reality reports are research diagnostics, not promotion evidence.

The existing `receipts/*.json` use several schemas. The current
`receipts-reverify` command reports unsupported receipts as failures and is
not a blocking CI gate until each class has a matching verifier.

See `MIGRATION.md` for the additive rollout phases.
