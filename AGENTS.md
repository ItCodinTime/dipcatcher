# AGENTS.md — working in this repo

## What this is

**fx-1** is the product: a quant research LLM fine-tuned from Kimi K3 open
weights. **dipcatcher** (`src/quant_fund`) is the harness: data engine,
evaluation, and verification that builds and gates fx-1 (`src/fx1`).

## Setup

```bash
make sync          # uv sync --frozen --all-groups (uv.lock is authoritative;
                   # --frozen must pass — regenerate with `uv lock` after
                   # touching [project])
```

Optional extras: `uv sync --all-groups --all-extras` adds torch (`nn` extra).
`MOONSHOT_API_KEY` is needed only for hosted eval (`make fx1-eval`); see
`.env.example` for all env vars.

## Gates (run before committing)

| Gate | Command |
|---|---|
| Lint | `make lint` (ruff check + format --check on `src`/`tests`) |
| Types | `make typecheck` (mypy `src/quant_fund`); fx1: `uv run mypy src/fx1` |
| Lab tests | `make test` (pytest: `tests/{unit,property,regression,end_to_end}`) |
| fx1 suite | `make fx1-test` — separate lane, ~200 tests |
| fx1 full gate | `make fx1-gate` — lint + types + tests + honesty + corpus smoke |
| CI parity | `.github/workflows/ci.yml` + `fx1.yml` use `uv sync --frozen` |

## Honesty contract (hard rules — do not weaken)

1. Research results are **proper scores** (pinball, CRPS, PIT, QLIKE, Brier,
   ECE, Kupiec, HMM likelihood). Never headline Sharpe/Sortino/Calmar/P&L/NAV.
   `FORBIDDEN_RESEARCH_METRIC_KEYS` in `quant_fund.research.catalog` is
   mirrored by `fx1.honesty.FORBIDDEN_HEADLINE_TOKENS` — change them together;
   `tests/fx1/test_honesty_inheritance.py` blocks drift.
2. SYNTHETIC results are correctness tests, always labeled, never market
   evidence.
3. No live-trading claims — no broker connectivity; see
   `docs/INSTITUTIONAL_READINESS.md` for the five minimum-evidence conditions.
4. Receipts are immutable evidence (`receipts/`, `verify-research`). Every
   research claim should be reproducible from a receipt hash.

## Conventions

- **Commits land directly on `main`** for fx-1 lanes; cross-cutting repo
  changes go through PRs. Match the existing terse conventional-commit style.
- `data/fx1/`, `data/*` derived dirs, `mlflow.db`, coverage files are
  gitignored — regenerate, never commit.
- `tests/fx1` is excluded from the default pytest testpaths on purpose; run it
  via `make fx1-test`. (The `tests/tests` mirror was dropped; if a stray
  reappears, keep it out of default collection.)
- Verifier acceptance history lives in `verifier/vN/`; run receipts in
  `verifier/runs/`. `INFLIGHT` / `day_grind_progress.md` / `docs/DATA_CONTRACTS.md`
  are the lab's work-tracking files — read them before grabbing work.
- `fx1_seed_corpus.jsonl` and the root docs (`APPLY.md`, `MATH_SPEC.md`,
  `RESEARCH_REFERENCES.md`) are tracked inputs/notes — don't delete blindly.

## Key docs

`docs/FX1.md` (model), `docs/FX1_TRAINING.md` (compute ladder + gates),
`docs/FX1_API_STABILITY.md` (versioning — `fx1.__version__` is canonical
semver; pyproject reads it via hatch dynamic version),
`docs/FX1_DATA.md`, `docs/FX1_DATASOURCES.md`, `docs/FX1_ARCHITECTURE.md`,
`docs/OPERATIONS_RUNBOOK.md`.
