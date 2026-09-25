# Verifier index — repo turnover to fx-1

Append-only index of acceptance-criteria versions. Each version lives in
`verifier/vN/acceptance.md`; every verification run is logged under
`verifier/runs/`.

| Version | Created (UTC) | Measures | Differs from prior |
|---|---|---|---|
| v1 | 2026-09-25 | Repo identity = fx-1 (pyproject, README, CLI); dipcatcher positioned as harness (`fx1.harness`); lab honesty gates untouched; fx1 test suite + ruff green | initial |
| v2 | 2026-09-25 | v1 + full 23-command harness registry, multi-source corpus (notebooks/ledgers), eval task bank, model cards + ship gate, inference backends, expanded CLI, Makefile targets | deeper coverage of turnover; 45 tests |
| v3 | 2026-09-25 | v2 + data quality gates, traces, DPO pairs, training receipts, statistical ship gate, cluster specs, staged pipeline, CI, tracking | industry-grade machinery; 56 tests |
| v4 | 2026-09-25 | v3 + hypothesis traces, reward model, curriculum, red-team suite, Dip Quality Score bench, cited serving, CLI additions | research-loop + flagship bench; 68 tests |
| v5 | 2026-09-25 | uniqueness deep research: audit + 6-domain landscape + 4 sequenced moves + boundaries; standalone report deliverable | research deliverable, not code |
| v5 | 2026-09-25 | four uniqueness moves: leakage-proof eval (masking/memory-gap/time-partition/contamination audit), attested inference (signing/TEE/zkML), hash-chained corpus ledger, MRM dossier | 84 tests |
| v6 | 2026-09-25 | auditor-grade: mypy gate, e2e lifecycle test, hypothesis property tests, SECURITY.md, architecture doc, API stability policy, SBOM generation | 91 tests, triple gate |
