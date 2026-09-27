# Contributing

**fx-1** is the model; **dipcatcher** (`src/quant_fund`) is the harness that
builds, evaluates, and verifies it. Read `AGENTS.md` first — it documents the
gates, the honesty contract, and the lab's work-tracking conventions.

## Before opening a PR

```bash
make sync        # locked env (uv sync --frozen --all-groups --all-extras)
make lint        # ruff check + format --check on src/ and tests/
make typecheck   # mypy src/quant_fund — add `uv run mypy src/fx1` for fx1 work
make test        # lab suite — or `make fx1-gate` for the full fx-1 CI lane
```

## Non-negotiables

- Research claims are proper scores only (pinball, CRPS, PIT, QLIKE, Brier,
  ECE, Kupiec, HMM likelihood) — never Sharpe/P&L/NAV headlines.
- SYNTHETIC results stay labeled SYNTHETIC; nothing is presented as live
  performance.
- `uv.lock` is authoritative — regenerate via `uv lock` after touching
  `[project]`; `uv lock --check` must pass.
- Don't commit derived artifacts (`data/fx1/`, coverage, `mlruns/`) or
  secrets (see `.env.example` for the env-var surface).
- fx1's forbidden-token set must stay mirrored with the lab catalog —
  `tests/fx1/test_honesty_inheritance.py` enforces it.
