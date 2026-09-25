# fx-1

**fx-1** (always lowercase) is a quant research LLM fine-tuned from the
open-weight **Kimi K3** base (`moonshotai/Kimi-K3`, 2.8T total / 104B active
MoE, Kimi K3 License). What makes fx-1 unlike any other model: it is trained
exclusively on **gate-passed, receipt-bound research behavior** — every
training example carries the SHA-256 of the verified artifact it came from,
and gate rejections are included as negative examples teaching refusal and
honest reporting.

**dipcatcher is the harness.** The lab formerly known as the project itself
now serves as fx-1's machinery:

| Harness role | dipcatcher capability |
|---|---|
| **Data engine** | Immutable receipts, benches, and ledgers become fx-1's training corpus (`fx1.data`) |
| **Evaluation** | Proper-score benches (pinball, CRPS, PIT, QLIKE, Brier/ECE, Kupiec, HMM likelihood) plus `fx1.eval` score the model's domain competence and behavior |
| **Verification** | `dipcatcher verify-research` and `dipcatcher doctor` gate every artifact fx-1 is trained on and every claim it makes |

See `docs/FX1.md` for the model architecture and `docs/FX1_TRAINING.md` for
the compute ladder (proxy → full-K3 LoRA → distilled student).

## The honesty contract (inherited by fx-1)

The harness enforces, and fx-1 is trained and tested to obey:

- Research results are **proper scientific scores** — never Sharpe, Sortino,
  Calmar, P&L, or NAV headlines.
- SYNTHETIC results are correctness tests, always labeled, never presented as
  market evidence.
- No live-trading claims: there is no live broker connectivity, and live
  readiness stays blocked until the five minimum-evidence conditions in
  `docs/INSTITUTIONAL_READINESS.md` are met.
- Fail-closed promotion: synthetic evidence, missing metrics, non-finite
  metrics, and invalid receipts cannot promote.

## Install

Python 3.12 and uv:

```
uv sync
```

## Quick start

fx-1 lifecycle:

```
fx1 corpus build                       # receipts -> SFT corpus (JSONL, provenance-hashed)
fx1 harness list                       # lab commands fx-1 may invoke
fx1 harness run verify-research        # verify harness artifacts
```

Training requires a corpus and a base-model eval on record first — the
manifest builder fails closed without them:

```
make fx1-corpus                        # -> data/fx1/corpus.jsonl
make fx1-eval                          # -> data/fx1/eval.json (needs MOONSHOT_API_KEY)
cp configs/fx1_run.example.json fx1_run.json   # fill in run_name + cost estimates
fx1 train manifest --config fx1_run.json       # immutable training-run manifest
```

Professional datasources (18 sources, all installed finance plugins):

```
fx1 sources list                       # registry + live availability probes
fx1 sources route --question "贵州茅台最新股价"   # authority-ordered candidates
fx1 sources fetch wind --api get_stock_price_indicators \
    --params-json '{"windcode":"600519.SH","indexes":"最新成交价"}' --as-of 2026-09-25
fx1 corpus ingest-source cls --api cls_telegraphs --as-of 2026-09-25 \
    --params-json '{"pageSize":50}'    # fetch -> gate -> corpus -> hash-chained ledger
```

Harness (dipcatcher) quick start:

```
uv run dipcatcher doctor
uv run dipcatcher research --config configs/research.yaml
uv run dipcatcher northset --config configs/research.yaml
uv run dipcatcher verify-research
```

`dipcatcher ingest` also writes `data/metadata/data_manifest.json` with source
labels, schemas, row counts, and SHA-256 hashes for the bronze/silver data
lake; `dipcatcher doctor` checks that manifest and the latest research
receipt before operators trust local state. Public/open feeds are collected
explicitly (network access is opt-in, never part of normal ingest):

```
uv run dipcatcher collect --source binance --param symbol=BTCUSDT --param interval=1d
uv run dipcatcher collect --source fred --param series_id=GDP
```

## Tests

```
uv run pytest
uv run ruff check src tests
uv run mypy src/fx1 src/quant_fund
```

## API security (harness service)

The FastAPI service (`dipcatcher api`, default host `127.0.0.1`):

- Config paths are allowlisted to the repo `configs/` directory.
- If `QUANT_API_KEY` is set, non-public routes require header `X-API-Key`;
  if unset, only loopback clients are accepted.
- Docker binds `127.0.0.1` by default; to listen on `0.0.0.0`, set
  `QUANT_API_KEY` and override the host.
- `POST /backtest` is deliberately bounded to 30 causal decision dates and at
  most 31 bar dates; run larger studies through the CLI.

## Documentation map

| Doc | Content |
|---|---|
| `docs/FX1.md` | fx-1 model: architecture, corpus provenance, honesty contract, license tier |
| `docs/FX1_TRAINING.md` | Compute ladder, pipeline contract, ship gate |
| `docs/RESEARCH_CENTRE.md` | Harness benches and research centre |
| `docs/INSTITUTIONAL_READINESS.md` | Evidence-gated readiness matrix (fx-1 inherits it) |
| `docs/REPO_IMPROVEMENT_PLAN.md` | Evidence-chain sequence |
| `docs/VALIDATION.md` | Walk-forward / CPCV / multiple-testing / conformal gates |
| `docs/OPERATIONS_RUNBOOK.md` | Operator procedures and incident handling |
| `docs/NORTHSET.md` | Order-book and candlestick slice |
| `docs/FX1_DATASOURCES.md` | 18 professional datasources (Wind/iFinD/Gildata/S&P/EDGAR/Yahoo/东财/CLS/财新/Binance/IMF/WB/IGO/XHCJ/research/天眼查/finance-fetch/Finenter): routing, honesty gates, PIT discipline |

*fx-1 and the dipcatcher harness produce research, backtest, or simulated
evidence only. Nothing here is investment advice or a promise of live profit.*
