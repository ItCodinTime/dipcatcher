# Artificial Hedge · Dipcatcher

Scientific **hedge research lab** for Artificial Hedge. Point-in-time benches for:

- stock ranking and residual alpha
- volatility (QLIKE)
- probabilistic return distributions (pinball, CRPS, PIT)
- market regimes (HMM likelihood)
- tail risk (VaR/ES, Kupiec)
- drawdown probability (Brier, log-loss, ECE)
- liquidity / implementation shortfall (Almgren–Chriss)
- reinforcement learning as a contextual bandit (LinUCB regret)

It is not a BUY/SELL LLM and not a Sharpe factory. Lab scores are proper scientific rules. SYNTHETIC oracle recovery is a correctness test, not live performance.

See [docs/RESEARCH_CENTRE.md](docs/RESEARCH_CENTRE.md).

## What it does not do

- Headline Sharpe, PSR, DSR, or live P&L as research output
- Same-close fills by default (see ADR-005)
- Silently relax infeasible optimizer constraints
- Auto-flatten the book on exceptions
- Present synthetic-market results as live evidence

## Install

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Quick start

```bash
uv run quant doctor
uv run quant research --config configs/research.yaml
```

`quant lab` is an alias. The notebook is `data/metadata/research/latest.md`.

## Tests

```bash
uv run pytest
uv run ruff check src tests
uv run mypy src/quant_fund
```

Default mode is `research`. Live trading requires explicit flags.
