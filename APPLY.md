# fx-1 overlay (v7 — datasource layer) — how to apply

Copy into your dipcatcher checkout, preserving paths:

    cp fx1_overlay/pyproject.toml fx1_overlay/README.md fx1_overlay/Makefile fx1_overlay/SECURITY.md <repo>/
    cp -r fx1_overlay/src/fx1 <repo>/src/
    cp -r fx1_overlay/tests/fx1 <repo>/tests/
    cp fx1_overlay/docs/FX1*.md <repo>/docs/
    cp -r fx1_overlay/.github/workflows/fx1.yml <repo>/.github/workflows/
    cp -r fx1_overlay/verifier <repo>/verifier

Verify: `uv sync && uv pip install -e .` → `uv run pytest tests/fx1`
(expect 137 passed) → `uv run ruff check src/fx1 tests/fx1` →
`uv run mypy src/fx1 --ignore-missing-imports` (all three clean).

Datasource layer needs the finance plugins installed (default root
`/app/.agents/plugins`, override with `FX1_PLUGIN_ROOTS`) and agent-gw
credentials in `KIMI_API_KEY` / `AGENT_GW_TOKEN` or `~/.kimi/agent-gw.json`.
Then: `fx1 sources list` → `fx1 sources describe wind` →
`fx1 sources fetch … --as-of YYYY-MM-DD`.

## Full contents (v1–v7)

- **Turnover**: fx-1 identity (pyproject/README/CLI), dipcatcher as harness.
- **Harness**: 23-command fail-closed registry, four roles.
- **Data**: receipts/notebooks/ledgers/traces corpora, quality gates,
  frozen splits, hash-chained corpus ledger.
- **Datasources (v7)**: 18 professional sources behind one honesty-gated
  adapter layer — agent-gw / custom-CLI / scenario-router / MCP kinds,
  credential gating (env-only, name-only probes), authority-ordered market
  routing with recorded honest degradation, PIT `as_of` ingest gate,
  live-claim quarantine to negative examples, hash-chained ledger
  provenance. CLI: `fx1 sources …` + `fx1 corpus ingest-source`.
  Verified live against CLS, Binance, IMF (see verifier/runs v7).
- **Eval**: task bank, red team, masked twins + memory gap, time partitions,
  contamination audit, statistical ship gate.
- **Training**: LoRA ladder, DPO pairs, immutable receipts, cluster specs,
  six-stage gated pipeline, tracking, curriculum.
- **Model**: versioned cards with ship gate, hosted-K3 + local backends.
- **Bench**: Dip Quality Score flagship.
- **Reward**: deterministic auditable reward model.
- **Uniqueness moves**: attested inference (sign/TEE/zkML), MRM dossier.
- **Auditor grade (v6)**: mypy gate, end-to-end lifecycle test, hypothesis
  property tests, SECURITY.md, architecture doc, API stability policy,
  hash-pinned SBOM generation (`fx1 sbom`).

Triple gate green: 137 tests, ruff, mypy. Verifier trail v1–v7 + runs/.
