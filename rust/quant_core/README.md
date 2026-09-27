# quant_core

Optional PyO3 extension for the numeric kernels that dominate research and
backtest *infrastructure* time. It is not an order router and it does not
touch live trading.

NumPy / the existing library functions stay the default when this extension
is not installed. `QUANT_FUND_NATIVE` is read once, at import of
`quant_fund.native`:

| value | behavior |
|---|---|
| `auto` (default) | use this extension when `import quant_core` succeeds, else NumPy |
| `python` | always NumPy / the library reference |
| `rust` | require this extension (`ImportError` if it is missing) |

```bash
make native
# equivalent:
# uv pip install "maturin>=1.7,<2"
# uv run maturin develop --release --manifest-path rust/quant_core/Cargo.toml
```

Existing CI jobs do not install Rust. The `rust-accel` job builds this crate
on Linux for CPython 3.12 and 3.13 and runs the parity tests.

## Kernels

Selected from a `time.perf_counter` survey of the research/backtest numeric
paths on this machine (CPython 3.12.3, NumPy 2.5.3). PR #99's suite is not on
`main`; the same idea — fixed SYNTHETIC inputs, median wall time — was used
here. These figures are infrastructure timings, not research scores.

| path | workload | median |
|---|---|---:|
| `indicators.bollinger` | n=80,000, window=20 | 431 ms |
| `indicators.rsi` | n=250,000 | 122 ms |
| `book_metrics_from_snapshot` | 2,000 books, depth 8 | 99 ms |
| `canonical_frame_fingerprint` | 10,000 rows | 78 ms |
| `indicators.ema` | n=250,000 | 54 ms |
| `apply_cross_sectional` | 200 names × 252 days | 36 ms |
| `hash_bytes` × 50,000 | 64-byte blobs | 18 ms |

`apply_cross_sectional` stayed in Polars: a second implementation would risk
the feature frame. The fingerprint's JSON canonicalization also stayed in
Python so receipt bytes do not change. The digest itself is SHA-256 and is
bit-for-bit with `hashlib`. Cumsum rolling mean/std are included because
feature code calls them directly and they match `lightspeed.ema` bit-for-bit.

Public signatures live on `quant_fund.native` and mirror the library
functions (`rolling_mean`, `rolling_std`, `ema`, `rsi`, `bollinger`,
`simple_returns`, `wealth_index`, `turnover`, `turnover_series`,
`book_features`, `hash_bytes`, `hash_many`).

## Float contract

Bit-for-bit with the NumPy/library reference, including NaN and inf:
`rolling_mean`, `rolling_std`, `simple_returns`, `wealth_index`, `hash_bytes`,
`hash_many`. Rolling moments use the same left-to-right prefix sums as
`numpy.cumsum`.

Within `quant_fund.native.reference.RTOL` / `ATOL` (`1e-12` / `1e-12`):
`ema`, `rsi`, `bollinger`, `turnover`, `turnover_series`, `book_features`.
NumPy's pairwise reductions (mean seeds, `std`, `sum`, `dot`) are not the
sequential `f64` sum this crate uses. Empty inputs, non-finite book rows, and
windows that do not fit the series are covered by the property tests.

Invalid book rows are NaN across every feature (fail-closed) instead of
raising, so a panel can be scored in one call. `book_metrics_from_snapshot`
still raises on a crossed or empty book; the parity tests compare only
valid snapshots.

## Speedup

SYNTHETIC wall clock on this VM. Seven timed repeats after one warmup, GC
disabled during the loop. Variance is the sample variance of those repeats
(seconds squared). Not a research score. `live_pnl_claim` is false.

Workloads: rolling/EMA n=40,000 window=20; RSI window=14 on the same series;
Bollinger window=20; simple returns and wealth n=80,000; turnover series
8,000×32; 4,000 books of depth 8; one 16 MiB SHA-256; 20,000×64-byte digests.

| kernel | Python median (ms) | Python variance (s²) | Rust median (ms) | Rust variance (s²) | speedup |
|---|---:|---:|---:|---:|---:|
| rolling_mean | 0.169 | 3.57e-9 | 0.128 | 1.68e-10 | 1.32× |
| rolling_std | 0.407 | 4.28e-9 | 0.195 | 5.43e-11 | 2.09× |
| ema | 8.555 | 1.03e-8 | 0.077 | 9.72e-11 | 111× |
| rsi | 17.287 | 5.07e-8 | 0.651 | 1.07e-9 | 26.5× |
| bollinger | 206.322 | 5.64e-7 | 0.957 | 1.63e-9 | 216× |
| simple_returns | 0.157 | 8.50e-9 | 0.067 | 1.32e-9 | 2.36× |
| wealth_index | 0.210 | 2.57e-9 | 0.183 | 1.57e-10 | 1.15× |
| turnover_series | 42.357 | 8.11e-9 | 0.234 | 3.57e-9 | 181× |
| book_features | 179.248 | 7.61e-7 | 1.177 | 3.92e-10 | 152× |
| hash_bytes | 8.327 | 3.77e-9 | 8.302 | 3.71e-9 | 1.00× |
| hash_many | 8.342 | 7.34e-9 | 3.368 | 1.87e-8 | 2.48× |

`hash_bytes` ties `hashlib` (OpenSSL) on a 16 MiB buffer. The win on hashing
is many small digests. Rolling mean/std and wealth are already C in NumPy, so
the speedup there is small; the Python window loops and the per-snapshot book
loop are the large ones.

Reproduce:

```bash
uv run pytest tests/native/test_native_bench.py -q -s
```

The crate is safe Rust: there are no `unsafe` blocks. `cargo fmt` and
`cargo clippy -- -D warnings` are clean on 1.83.0.
