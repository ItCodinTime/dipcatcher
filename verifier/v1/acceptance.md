# Acceptance criteria v1 — repo turnover to fx-1

The repo is "turned over to fx-1 with dipcatcher as harness" iff ALL hold:

1. **Identity:** `pyproject.toml` project name is `fx-1`; description states
   fx-1 is the model and dipcatcher is the harness; wheel packages include
   both `src/fx1` and `src/quant_fund`; scripts expose `fx1` plus the
   harness aliases `dipcatcher`/`quant`.
2. **README:** `README.md` leads with fx-1 as the project and documents
   dipcatcher's harness role (data engine, eval harness, verification).
3. **Harness wiring:** `src/fx1/harness.py` exposes the lab's benches and
   verification as typed harness commands with runner injection (no hidden
   subprocess in tests); `src/fx1/cli.py` provides the `fx1` CLI with corpus,
   eval, train-manifest, and harness commands.
4. **Gates untouched:** no edits to `quant_fund` honesty/promotion logic;
   `FORBIDDEN_RESEARCH_METRIC_KEYS` and fail-closed gates unchanged.
5. **Quality:** `pytest tests/fx1` 100% pass; `ruff check src/fx1 tests/fx1`
   clean.

Verification commands are logged per run in `verifier/runs/`.
