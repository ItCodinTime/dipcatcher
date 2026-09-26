# Hugging Face OHLCV-1m

Research-only minute bars from
[mito0o852/OHLCV-1m](https://huggingface.co/datasets/mito0o852/OHLCV-1m).
This is not a SIP feed, not a corporate-action vintage, and not a live-trading
source. Nothing here is a performance claim.

## What was on the Hub

Inspected 2026-09-26 against commit
`776328445b7ac6e7815ef3a483e9c8ded1eb6d56` (card `lastModified`
2026-05-03, created 2025-07-14). The dataset is public and not gated.

| Item | Finding |
|---|---|
| Layout | 411 files, `data/ohlcv_YYYY-MM.parquet`, January 1992 through March 2026. No other data files. |
| Size | 87,719,544,647 bytes across those files (~81.7 GiB). The card's "87.7 GB" matches. Individual months run from ~22 MB (1992) to ~497 MB (`ohlcv_2021-03.parquet`). |
| License | **None declared.** `cardData` has no `license` field, the tag list is only `region:us`, and the repo has no `LICENSE` file. The card says the bars were aggregated from Finnhub.io monthly archives. This repository does not vendor the corpus and does not grant redistribution rights. |
| Card vs files | The dataset viewer fails with `FileFormatMismatchBetweenSplitsError`. The README YAML lists per-ticker splits (`data/AOM-*`, …) that are **not** in the file tree, and a `train` split of 3,538,138 rows / ~196 MB. January 1992 alone is 4,609,974 rows. Trust the Parquet files, not that YAML. |
| Coverage end | Files stop at March 2026. The card's "1992–2026" is not a live tail. |

Vendor schema (confirmed on `ohlcv_1992-01.parquet`, 33,507,926 bytes, 5 row groups):

| Column | Type | Meaning |
|---|---|---|
| `timestamp` | `datetime[ns, UTC]` | Start of the minute. Second and sub-second are 0 in the inspected month. |
| `open`, `high`, `low`, `close`, `volume` | `float64` | Volume is a share count stored as float. |
| `ticker` | `string` | US ticker, uppercase in the sample, including dotted preferreds (`AA.PR`) and warrants (`AAC.WS`). |

January 1992, read in full:

- 6,338 tickers, 4,609,974 rows, no nulls, no duplicate `(ticker, timestamp)`.
- Timestamps run `1992-01-02 14:30:00Z` through `1992-01-31 22:14:00Z`. In America/New_York that is 09:30–17:26, weekdays only. No pre-market prints in this month. 4,488,728 rows are regular hours; 121,246 are at or after 16:00.
- `14:30Z` is 09:30 EST, so `timestamp` is the **open** of the minute, matching the card.
- Prices are positive. The minimum print is `0.00781` (a 1/128). Eighths and sixteenths are common (AAPL opens 1992-01-02 at 55.75).
- One envelope break: `TWX` at `1992-01-15 21:05:00Z` has `open=95` and `low=high=close=95.25` (`open < low`). Zero rows with `high < low` in this month.
- Minutes are trade prints, not a filled clock. AAPL has 7,429 rows in the month and 1,052 intraday gaps longer than one minute (max 77). On 1992-01-02 AAPL has 382 prints, not 390, with a 15-minute hole.
- Row groups mix many tickers. A symbol filter still has to read the month file. The cache granule is one month, not one ticker.
- Adjustment is **not stated**. A full-file check of a later split (for example AAPL in August 2020, ~309 MB) was not required to land the adapter; do not assume the series is split-adjusted or raw. There is no corporate-action column.

## Schema mapping

| Vendor | Canonical bronze | Rule |
|---|---|---|
| `ticker` | `security_id`, `symbol` | Uppercased, not otherwise rewritten. Dots stay. There is no other stable id. |
| `timestamp` | `event_time`, `available_time` | Both equal `timestamp + 1 minute` (UTC close). This matches the bar-close PIT convention. It is not a SIP publication time. |
| OHLC, `volume` | same names | Unchanged. Invalid rows raise; they are not clamped. |
| — | `source` | `hf_ohlcv_1m` |
| — | `revision_id` | `HF_OHLCV_1M_UNDECLARED_ADJ` |
| — | `currency` | `USD` assumption. The file has no currency column. |
| — | `session` | From the minute **open** in America/New_York. `rth`: weekdays, 09:30 inclusive through 16:00 exclusive. `ext`: weekdays, 04:00–09:30 or 16:00–20:00. `off`: anything else, kept and counted. |
| — | `ingested_time` | Wall clock at normalization. |
| — | corporate actions | Empty. Do not invent splits. |
| — | security master | `exchange=UNKNOWN`, `security_type=unknown`, `valid_from` = first print, `valid_to` null. |

`dipcatcher collect` writes that frame under `data/raw/sources/` with the usual JSON receipt. Ingest from `data.source: hf_ohlcv_1m` reads the **local cache only** and does not download. Point `data.source_path` at the cache directory and set `data.source_symbol` to a US ticker (`source_interval` selects `1m`, `5m`, `15m`, `30m`, `1h`, or `1d`).

Date bounds: a `YYYY-MM-DD` value is an America/New_York calendar day (inclusive). Aware datetimes are UTC bounds on `event_time`. Naive datetimes raise.

## Quality rules

Failures raise `OhlcvQualityError` and leave the rows unchanged:

- duplicate `(ticker, timestamp)` or `(security_id, event_time)`
- naive or non-datetime timestamps, or timestamps that are not minute-aligned
- non-positive or non-finite OHLC, `high < low`, open/close outside `[low, high]`
- negative or non-finite volume
- a requested symbol with no rows in the selected months

Missing minutes are flagged, not filled. `minute_gap_report` compares each observed New York session date with the 390 regular-session minute starts. `strict_gaps=True` raises when any of those minutes are absent (early closes and holidays will trip it: `session_days` is weekdays only). `strict_off=True` raises on `off` session rows instead of dropping them. The default is to return the report on `provider.quality_report` and keep the prints.

Resample with `resample_ohlcv` (also `interval=` on the provider). There is no other clock-time OHLCV aggregator in the repo. Northset `session_candles_from_daily` builds a synthetic path from a daily envelope and is not used. Intraday bins align to America/New_York, with hourly bins offset to 09:30. Daily bars group by the New York date, not UTC midnight. Empty bins are not inserted. `n_source_minutes` is the number of observed minutes, and a bin that mixes sessions is labeled `mixed`. `event_time` remains the close of the last observed minute.

Downloads are capped at six monthly files unless `max_months` is raised, and each file at 1 GB. The default cache is `data/hf_ohlcv_1m/` (`HF_OHLCV_1M_CACHE`). Files are stored under the pinned revision id. A UTC window that crosses a month boundary reads every month the vendor timestamps touch; a 404 fails that request rather than silently omitting the boundary.

## Use

```bash
uv run dipcatcher collect --source hf_ohlcv_1m \
  --param symbols=AAPL,MSFT \
  --param start=2024-01-02 \
  --param end=2024-01-05 \
  --param interval=5m
```

```python
from quant_fund.data.adapters.hf_ohlcv_1m import HfOhlcv1mProvider, minute_gap_report

provider = HfOhlcv1mProvider(
    "data/hf_ohlcv_1m",
    symbols="AAPL",
    start="2024-01-02",
    end="2024-01-05",
    allow_download=True,
)
bars = provider.get_bars()
gaps = minute_gap_report(bars)
```

Those bars satisfy the bronze contract, so `adjust_prices`, `canonical_northset_bars`, and `bench_candle_order_book` can take them without a separate schema. Pass `northset.require_adjusted_ohlc=false` when you still hold raw bronze. After `adjust_prices` on the empty action table, silver `split_factor` is 1.0 and Northset will stamp `price_basis=split_adjusted`. That stamp means "identity factor", not "splits were removed".

Candle/book benches that are not given an external book still synthesize L2 (`book_source=synthetic_lob`). The minute bars do not become a vendor order book.

## Tests

Offline tests read `tests/fixtures/hf_ohlcv_1m/ohlcv_sample.parquet`. The optional Hub check is marked `network` (CI runs `pytest -m "not network"`).
