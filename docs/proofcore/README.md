# PROOFCORE

**PROOFCORE** is an experimental set of PIT storage, proof foundations,
leakage checks, and statistical diagnostics layered above the existing
research packages. Bundle hashes, sidecars, signatures, and metrics can be
checked now. Historical proven-run orchestration and replay remain closed
until a decision schedule drives actual point-in-time vault reads.

## What it adds

| Component | Package | What it does |
|---|---|---|
| Shared contracts | `quant_fund.proofcore.contracts` | pydantic schemas (`ProofBundleV1`, `TrialLedgerRow`, `LeakageReport`, `RealityReport`), canonical hashing, error taxonomy |
| PIT Vault (W1) | `quant_fund.pit` | Write-once, content-addressed, bitemporal store; `asof(t)` is the ONLY legal read path |
| Proof foundations (W2) | `quant_fund.proof` | Bundle construction, signing, and verification of hashes, sidecars, and recomputed metrics; historical run and replay entry points fail closed |
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

# verify an existing bundle and its chain (requires the HMAC key for signed bundles)
quant proof verify --bundle path/to/bundles/id.json --bundle-dir path/to/chain
# quant proof run and quant proof verify --replay fail closed pending causal reads

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
make proof-verify         # bundle verifier tests; no historical replay claim
make leakage-scan         # warn mode this wave (adjudicated)
make reality-gate         # score trials; absent proofcore.duckdb prints REALITY_FILTER_SKIP and exits 0
make receipts-reverify    # fail-closed audit; heterogeneous receipt verifiers pending
```

## Honesty contract

PROOFCORE follows the AGENTS.md honesty rules. The standalone verifier checks
bundle integrity and recomputes metrics from trade-log bytes. It does not
establish that precomputed weights were available to each historical decision;
the proven runner and replay therefore fail closed. The provenance DB logs
bundles as unverified and rejects `--verification` until a bound result schema
and ingestion path are implemented. Reality reports are research diagnostics,
not promotion evidence.

The existing `receipts/*.json` use several schemas. The current
`receipts-reverify` command reports unsupported receipts as failures and is
not a blocking CI gate until each class has a matching verifier.

See `MIGRATION.md` for the additive rollout phases.
