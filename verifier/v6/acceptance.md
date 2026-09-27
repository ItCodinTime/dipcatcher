# Acceptance criteria v6 - auditor-grade completeness

Extends v5 (all prior criteria hold). New:

1. **Triple gate:** pytest (tests/fx1) + ruff + mypy all clean on src/fx1.
2. **End-to-end integration test:** full lifecycle receipts->corpus->quality->
   ledger->split->eval->receipt->compare->card->signed release->serving->MRM
   dossier in one test.
3. **Property-based tests (hypothesis):** honesty validator never crashes;
   masking idempotent; reward bounded; dip events causal/bounded — for
   arbitrary inputs.
4. **SECURITY.md:** scope, hard rules, reporting, supply-chain posture.
5. **Architecture doc** with trust flow + invariant table (FX1_ARCHITECTURE.md).
6. **API stability policy** (FX1_API_STABILITY.md): stable/unstable surfaces.
7. **SBOM:** hash-pinned generation from uv.lock; fail-closed; CLI `fx1 sbom`.
