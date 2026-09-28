# Demo data — offline end-to-end bootstrap

A fresh clone can run the ingest → features → backtest chain without vendor
keys or large downloads. `make demo-data` writes a small, deterministic,
**labeled-SYNTHETIC** dataset into `data/demo/` (gitignored — the generator is
the artifact). Every canonical row carries `source="synthetic"` /
`revision_id="SYNTHETIC"`, so downstream honesty labels (`SYNTHETIC` echoes,
`source_note`, `research_only`, `live_pnl_claim=false`) propagate normally.

> Results on demo data prove **pipeline correctness only — never market
> evidence**. Do not cite demo metrics as research results.

## Generate

```bash
make demo-data          # uv run python scripts/gen_demo_data.py
# or: uv run python scripts/gen_demo_data.py --out data/demo --seed 13 --days 140
```

Layout written under `data/demo/`:

| Path | Contents |
|---|---|
| `daily/raw/bars.parquet` | 8 symbols (MKT + 7 names) x 140 daily bars, bronze bar contract + `planted_signal` oracle column |
| `daily/raw/corporate_actions.parquet` | 1 split (2:1 on SEC_0001) + 1 cash dividend (SEC_0002) |
| `daily/raw/security_master.parquet` | 8 security-master rows |
| `minute_cache/<hf-rev>/ohlcv_YYYY-MM.parquet` | vendor-shaped 1-minute slice: 2 tickers x 3 sessions x 390 RTH minutes |
| `manifest.json`, `README.md` | seed, row counts, sha256 digests, SYNTHETIC banner |

Determinism: same `--seed` produces identical rows (byte-identical parquet in
the same environment) — `ingested_time` is pinned to last-close + 1h, never
wall clock.

## Daily bars: feature build + backtest + receipt

```bash
uv run dipcatcher ingest --config configs/demo.yaml           # bronze->silver + data_manifest.json
uv run dipcatcher build-features --config configs/demo.yaml   # gold/features.parquet + gold/labels.parquet
uv run dipcatcher backtest --config configs/demo.yaml         # prints SYNTHETIC + metrics
```

Persisted receipt (analytics export, research-only stamped):

```bash
uv run python - <<'PY'
from quant_fund.backtest.engine import export_backtest_metrics_json, run_backtest
from quant_fund.config import load_config
from quant_fund.features.engine import build_features
from quant_fund.pipeline.dataset import ensure_silver
from quant_fund.pipeline.forecast import build_causal_weight_panel

cfg = load_config("configs/demo.yaml")
bars = ensure_silver(cfg)
feat = build_features(bars, cfg)
weights = build_causal_weight_panel(cfg, feat["event_time"].unique().sort().to_list())
result = run_backtest(feat, weights, cfg)
print(export_backtest_metrics_json(result, cfg.data.root / "metadata" / "backtest_receipt.json"))
PY
```

The receipt JSON stamps `source_note="SYNTHETIC"`, `research_only=true`,
`live_pnl_claim=false`, and `analytics_export_sha256`.

## Minute bars

```bash
uv run dipcatcher ingest --config configs/demo_minute.yaml    # hf_ohlcv_1m provider, offline
```

`data/demo/minute/{bronze,silver,metadata}` is a separate lake root so the
daily lake is not clobbered. `allow_download` stays off — the provider only
reads the generated month cache.

## Config notes

`configs/demo.yaml` sets `data.source: file` with `data.root: data/demo/daily`.
Two universe floors are relaxed for the tiny panel so every silver session is
a usable decision date: `min_history_bars: 1` (the default 60 would eat the
140-session span) and `min_adv: 0` (trailing ADV needs a 20-bar warmup;
without this the first ~19 sessions have no universe members and the
`dipcatcher backtest` causal-weight loop fails closed on empty days — a real
fail-closed edge, worked around here deliberately for the demo).

## Honesty / limitations

- Every row is labeled SYNTHETIC; the backtest prints `SYNTHETIC` and the
  receipt stamps `research_only` / `live_pnl_claim=false`.
- `dipcatcher research`, `train`, and `forecast` label their notebooks by
  `config.data.source` (`"file"` for this demo) rather than row provenance, so
  prefer the backtest receipt above, which labels from row `source`. The
  ranking-side benches also want more history than 140 sessions provides.
- The daily panel mirrors `SyntheticMarketProvider` output with the
  deliberately delisted name dropped — a delist would strand an unvalued
  hold in the naive `dipcatcher backtest` demo path (see
  `tests/end_to_end/test_synthetic_pipeline.py` for the causal workaround).
- Demo data is a correctness fixture: no splits beyond one 2:1, one dividend,
  no delisting, no restatements, no real-world microstructure.

Tests: `uv run pytest tests/unit/data/test_demo_data.py -q` (generator
determinism, file/HF provider loads, ingest+gold, SYNTHETIC backtest label).
Committed fixture: `tests/fixtures/demo/` (6 bars + master slice).
