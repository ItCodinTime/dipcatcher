# Acceptance criteria v5 - four uniqueness moves implemented

Extends v4 (all prior criteria still hold). New:

1. **Leakage-proof eval:** deterministic masking (tickers/dates/years ->
   consistent placeholders; proper-score vocab preserved); masked twins;
   memory-gap ship metric with budget; time-partition (post-cutoff slice);
   contamination audit with 3 probes (n-gram, Min-K%, rephrased-gap), each
   with stated limitations, hash-bound report.
2. **Attested inference:** HMAC release signing (env-only key,
   Sigstore-compatible shape); tamper-evident verify; LocalFx1Backend refuses
   unsigned checkpoints when FX1_SIGNING_KEY set; TEE quote schema with
   checkpoint binding + nonce anti-replay; selective-zkML manifest requires
   proof artifacts for every covered operator; ladder status reporter.
3. **Corpus ledger:** hash-chained append-only entries (example/exclusion),
   rule-attributed exclusions, tamper-evident verify_chain, public
   audit_export with no raw data.
4. **MRM dossier:** five-activity compilation (development, implementation,
   validation, monitoring, governance), artifact-hash citations, fail-closed
   on missing artifacts, contamination flag surfaced, disclaimer embedded.
5. **CLI:** maskedaeval, contamination-audit, sign, attestation, mrm.
6. **Quality:** full suite green; ruff clean.
