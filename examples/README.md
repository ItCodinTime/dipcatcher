# Examples

Runnable research gallery. Each file is a [jupytext](https://jupytext.readthedocs.io/) percent-format script: `uv run python examples/<name>.py` executes it, and a notebook UI can open the same file. Nothing here is investment advice or a live-trading claim.

| Example | Data label | What it shows |
|---|---|---|
| `01_receipt_round_trip.py` | SYNTHETIC | Small synthetic research run, receipt verification, one-byte tamper rejected |
| `02_purged_walkforward_conformal.py` | tracked real snapshot | Purged/embargoed walk-forward, split-conformal intervals, pinball / CRPS / coverage. Skips when `data/file_us_wide/bronze/bars.parquet` is absent or its SHA-256 does not match `configs/real_benchmark_us_wide.json` |
| `03_synthetic_l2_book.py` | SYNTHETIC | Synthetic L2 book, VPIN proxy, queue imbalance, candle/book as-of join |
| `04_hf_ohlcv_1m.py` | fixture, or network | `hf_ohlcv_1m` on a checked-in fixture. `HF_OHLCV_1M_ALLOW_DOWNLOAD=1` fetches one monthly file for a one-day AAPL window into a temp cache |
| `05_phase1_evidence.py` | tracked real snapshot | `verify_phase1_index` / `verify_phase1_run` on the sealed Phase-1 evidence index |

```bash
uv run python examples/01_receipt_round_trip.py
make examples   # ruff, mypy, and the offline pytest runner
```

`make examples` runs each script in a subprocess with a 120 second timeout and with outbound sockets blocked. The default pytest lane does not collect `tests/examples/`.

## Labels

SYNTHETIC output is an engine check, not market evidence. The US snapshot scores are proper scores on a survivorship-biased vendor file whose availability timestamps were reconstructed; the example prints those disclosures from the benchmark config. Phase-1 verification can report `runtime differs from this environment` when this interpreter's patch version is not the one that sealed the receipts. That note is not a broken seal. Any other verifier error fails `05_phase1_evidence.py`. A verified receipt does not authorize live trading.

## Hugging Face OHLCV-1m

The Hub dataset `mito0o852/OHLCV-1m` declares no license (upstream Finnhub). This repo does not vendor it and does not grant redistribution rights. See `docs/HF_OHLCV_1M.md`. Adjustment is undeclared.

Offline runs copy `examples/fixtures/hf_ohlcv_1m/ohlcv_sample.parquet` (a few synthetic vendor-schema rows, not Hub data) into a temporary cache. A network run is opt-in, downloads a single monthly Parquet into a temporary directory, and leaves the repository unchanged.
