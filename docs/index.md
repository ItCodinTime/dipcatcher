# dipcatcher

**dipcatcher** (`quant_fund`) is the research harness for **fx-1**, a quant research LLM fine-tuned from Kimi K3 open weights. The harness is the data engine, the evaluation bench, and the receipt verifier. fx-1 is the model. This site is the reference for both.

The lab builds a point-in-time market-state object, scores forecasts with proper scoring rules, and keeps every claim tied to a sealed receipt. Simulated paper and shadow runs exercise those same gates. Live broker connectivity is not implemented.

## Honesty and receipts

Research results are proper scores:

- pinball
- CRPS
- PIT
- QLIKE
- Brier
- ECE
- Kupiec
- HMM likelihood

A research claim is reproducible from an immutable receipt. `dipcatcher verify-research` checks the seals — config hash, dataset bytes, git revision, package versions, and the benchmark catalog — and rejects a forged live-trading claim. Receipts live under `receipts/` and `data/metadata/research/`.

Sharpe, Sortino, Calmar, P&L, and NAV are excluded from research headlines. The token list is `FORBIDDEN_RESEARCH_METRIC_KEYS` in `quant_fund.research.catalog`, mirrored by `fx1.honesty.FORBIDDEN_HEADLINE_TOKENS`.

See [Receipt verification](RECEIPT_VERIFICATION.md) and [Institutional readiness](INSTITUTIONAL_READINESS.md).

## Real data and SYNTHETIC data

| Label | Meaning |
|---|---|
| **SYNTHETIC** | Simulator or planted-factor panel. A passing score is an engine correctness test, and the label stays on the receipt. |
| **public / file** | Local CSV or Parquet, including session-close tapes. Scores stay on proper rules. `available_time` is a publication convention, not a SIP vintage. |
| **public sources** | Registered feeds collected with `dipcatcher collect` (FRED, EDGAR, Treasury, CFTC, Binance klines, Hugging Face `hf_ohlcv_1m`, and others). Each collection writes a JSON receipt. |

SYNTHETIC evidence cannot take a champion or live alias. The full rules are in [Data source labels](DATA_SOURCE_LABELS.md). Minute bars from Hugging Face are specified in [HF OHLCV-1m](HF_OHLCV_1M.md).

## Where to go

- [Getting started](getting-started.md) — install, doctor, a SYNTHETIC research smoke, and this site
- [Architecture](ARCHITECTURE.md) — package layout and the decision clock
- [Data contracts](DATA_CONTRACTS.md) — point-in-time fields and lake tables
- [Validation](VALIDATION.md) — walk-forward, purge, embargo, CPCV
- [Mathematical specification](MATH_SPEC.md) — formulas the code implements
- [Research centre](RESEARCH_CENTRE.md) — benchmark families and what a lab headline is
- [fx-1](FX1.md) — the model, its corpus, and the honesty gate
- [API reference](api/index.md) — docstrings for the public harness modules
