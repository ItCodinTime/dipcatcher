# Artificial Hedge · Dipcatcher

Dipcatcher is **Artificial Hedge’s proprietary research lab**. It provides point-in-time benches for:

- stock ranking and residual alpha
- volatility (QLIKE)
- probabilistic return distributions (pinball, CRPS, PIT)
- market regimes (HMM likelihood)
- tail risk (VaR/ES, Kupiec)
- drawdown probability (Brier, log-loss, ECE)
- liquidity / implementation shortfall (Almgren–Chriss)
- reinforcement learning as contextual bandits (LinUCB, linear/quantile Thompson sampling, and neural policy gradient)
- **Northset** — order-book snapshots and candlesticks (identities, OHLC vol, Kyle λ, Roll, OFI, VPIN)

It is not a BUY/SELL LLM and not a Sharpe factory. Lab scores are proper scientific rules. SYNTHETIC oracle recovery is a correctness test, not live performance.

See [docs/RESEARCH_CENTRE.md](docs/RESEARCH_CENTRE.md) and [docs/NORTHSET.md](docs/NORTHSET.md).
For the full ML/RL training commands, artifacts, and evidence boundaries, see
[docs/ML_RL_CAPABILITIES.md](docs/ML_RL_CAPABILITIES.md).
For the evidence-gated production boundary, see
[docs/INSTITUTIONAL_READINESS.md](docs/INSTITUTIONAL_READINESS.md).
For operator procedures and incident handling, see
[docs/OPERATIONS_RUNBOOK.md](docs/OPERATIONS_RUNBOOK.md).
For a fixed dataset/split contract with matched forecast baselines, see
[docs/REAL_DATA_BENCHMARK.md](docs/REAL_DATA_BENCHMARK.md).
The implementation sequence is tracked in [docs/REPO_IMPROVEMENT_PLAN.md](docs/REPO_IMPROVEMENT_PLAN.md).

## What it does not do

- Headline Sharpe, PSR, DSR, or live P&L as research output
- Same-close fills by default (see ADR-005)
- Silently relax infeasible optimizer constraints
- Auto-flatten the book on exceptions
- Present synthetic-market results as live evidence

## Paper / shadow (Phase 17)

Simulated broker only — no live fills:

```bash
uv run dipcatcher paper --config configs/paper.yaml --max-steps 8
```

Ledgers land in `data/metadata/paper/`. SYNTHETIC runs are research/infrastructure only.

## Install

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Quick start

```bash
uv run dipcatcher doctor
uv run dipcatcher research --config configs/research.yaml
uv run dipcatcher northset --config configs/research.yaml
uv run dipcatcher verify-research
```

The notebook is `data/metadata/research/latest.md`. The `quant` executable remains a
backward-compatible alias for older workflows; new integrations should use `dipcatcher`.

`dipcatcher ingest` also writes `data/metadata/data_manifest.json`, including source labels,
schemas, row counts, and SHA-256 hashes for the bronze/silver data lake. `dipcatcher doctor`
checks that manifest and the latest research receipt before operators trust the local data state.

Public/open feeds are collected explicitly (network access is opt-in and never part of
normal ingest):

```bash
uv run dipcatcher collect --source binance --param symbol=BTCUSDT --param interval=1d
uv run dipcatcher collect --source fred --param series_id=GDP
```

Each collection lands under `data/raw/sources/` with a JSON receipt (SHA-256, row counts,
PIT ranges, provenance). Bar-capable sources can also be routed through `data.source` in
config (e.g. `binance_public_data`). See `docs/DATA_SOURCE_LABELS.md`.

The optional Kronos candlestick adapter (`train.kronos` config) is research-only, loads
strictly local pre-downloaded artifacts (`dipcatcher[nn]` extra), and never reaches the
network or live execution paths.

## Tests

```bash
uv run pytest
uv run ruff check src tests
uv run mypy src/quant_fund
```

Default mode is `research`. Live trading requires explicit flags.

## API security

The FastAPI service (`dipcatcher api`, default host `127.0.0.1`):

- Config paths are allowlisted to the repo `configs/` directory (path traversal rejected).
- If `QUANT_API_KEY` is set, non-public routes require header `X-API-Key`.
- If unset, only loopback clients are accepted — remote unauthenticated access is refused.
- Docker image binds `127.0.0.1` by default and syncs with `uv.lock` (`uv sync --frozen`). To listen on `0.0.0.0`, set `QUANT_API_KEY` and override the host.
- The HTTP `POST /backtest` route is deliberately bounded to 30 causal decision dates and at most 31 bar dates (the extra date preserves next-open execution). Run larger historical studies through the CLI rather than the API worker.

## Research validation

- `dipcatcher research` — scientific benches (proper scores; SYNTHETIC labeled)
- `dipcatcher northset` — order-book and candlestick slice (identities, OHLC vol, Kyle/Roll/OFI/VPIN)
- `dipcatcher validate <model_id>` — fail-closed causal / walk-forward / promotion gates
- `dipcatcher verify-research` — verify immutable provenance, scorecards, and research artifacts
- `configs/production.yaml` — **not** a live broker profile; research-strict promotion gates only (Phase-17 broker absent)
- See `docs/DATA_SOURCE_LABELS.md` and `docs/VALIDATION.md`
