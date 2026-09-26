# Data source labels

Dipcatcher never conflates lab correctness with live edge.

Configured data sources are explicit: `synthetic` uses the labeled simulator;
`file` and `parquet` use the local CSV/Parquet adapter. Unknown source
identifiers are rejected during configuration rather than silently routed to a
fallback adapter.

| Label | Meaning |
|---|---|
| **SYNTHETIC** | Planted-factor / simulator panel. Recovering IC, coverage, or bandit regret is an **engine correctness** test. Never a production promotion input. |
| **public / file** | Features built from the configured non-synthetic adapter (parquet, Stooq session-close tape, etc.). Scores are scientific (proper rules) still, not a live-P&L claim. Session-close `available_time` is not a SIP vintage. |
| **public sources** | Registered open/public feeds collected explicitly via `dipcatcher collect` (Binance klines, FRED/ALFRED, US Treasury, CFTC COT, FINRA short volume, World Bank, BEA, GDELT, SEC EDGAR, NASDAQ ITCH sample, FI-2010, Hugging Face `hf_ohlcv_1m`) or routed through `data.source` for bar-capable feeds. Point-in-time stamped (`event_time`/`available_time`/`ingested_time`, `source`, `revision_id`) with a per-collection JSON receipt (sha256, rows, provenance). Still research evidence — never a live-P&L claim. Optional-library feeds (`ccxt`, `cryptofeed`, `openbb`) require an explicit payload and are never imported implicitly. `hf_ohlcv_1m` adjustment is undeclared; see `docs/HF_OHLCV_1M.md`. |

## Where the label appears

- Research notebook JSON/Markdown (`data_source`, `synthetic`, disclaimer banner)
- `dipcatcher research` stdout (`SYNTHETIC` line when applicable); legacy `quant` executable alias
  remains callable as a compatibility alias
- `dipcatcher validate` report (`data_label`, `DATA_LABEL=SYNTHETIC`)
- Backtest metrics (`data_source`)
- MLflow tags / promotion gate (`synthetic_evidence_not_promotable`)

## Fail-closed rules

1. SYNTHETIC evidence cannot receive a champion / live alias.
2. `dipcatcher validate --claim-live` on SYNTHETIC fails (`ok=false`).
3. Missing causal weight panel **and** missing walk-forward / notebook evidence fails validation.
4. Sharpe / PSR / simulated P&L are not research-lab headlines (see `docs/RESEARCH_CENTRE.md`).

## Overnight note (Wave 20, 2026-09-16)

Paper / validate / research CLI banners surface these labels (`DATA_LABEL=SYNTHETIC`,
`live_pnl_claim=false`, `research_only`). Overnight paper smokes and promotion dry-runs
are **infrastructure correctness** only — never a live P&L claim. See also
`docs/PAPER_SHADOW.md` and `docs/VALIDATION.md`.
