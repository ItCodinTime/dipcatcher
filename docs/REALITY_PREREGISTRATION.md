# Reality-filter trial pre-registration

Frozen before any strategy result was computed. The machine-readable copy is
`research/reality/preregistration.json`. The runner must refuse a cost or
risk-gate config that differs from that file, and it must record every grid
cell, including cells that lose.

This is a research diagnostic. It is not a live-trading claim, not a
promotion, and not a broker test.

## Why these books

Two existing sleeves in `quant_fund.backtest.sleeves`, plus one unswept
baseline:

- `sweep_reclaim_weights` is the dip sleeve. A swept prior low that reclaims
  is a long; a swept prior high that reclaims is a short. The pulse decays
  over a fixed hold. That is the repo's dip-buying rule. The grid only varies
  lookback, hold, and decay.
- `slow_trend_weights` is the trend sleeve beside it. The sign of a shifted
  fast mean minus a shifted slow mean is inverse-vol sized and demeaned
  across the names that session.
- Equal-weight long is one cell, not a search. Each session's names share
  weight, clipped by the frozen risk-gate name cap and scaled to the frozen
  net cap. It is there so the ledger has three strategy clusters. The
  reality filter will not leave `insufficient_evidence` for cluster count
  until three clusters exist. Adding the cell does not change the filter
  thresholds.

No new signal is defined.

## Data

Public Yahoo daily bars through `fetch_yahoo_chart` /
`parse_yahoo_chart`, symbols `US_LIQUID` (sixteen current US names). No
API key. The Hugging Face OHLCV-1m tape is the other public source in the
repo; its files are full-market months, hundreds of megabytes each, and a
multi-year daily panel would mean downloading the whole tape. This study
uses the daily adapter instead. If the Yahoo fetch fails, the run stops.
Synthetic prices are not a fallback.

A name stays in the panel only if it has a New York session on or before
2014-06-30 and on or after 2024-06-28. Tickers are not added. Fewer than
eight surviving names aborts the run.

`US_LIQUID` is a present-day list, so names that did not survive are
missing. The adapter stores quote OHLC and stamps `YAHOO_VENDOR_ADJ`. The
mark used as `close_total_return` is that close. There is no separate
dividend series in the adapter. This is not a SIP as-of vintage.

Prices stay in `data/reality_sweep` and are not committed. The run records
the parquet hash.

## Splits

New York session dates, half-open:

| Window | Dates | Role |
|---|---|---|
| Train | 2016-01-04 ≤ d < 2020-01-01 | In-sample diagnostics only |
| Validation | 2020-01-02 ≤ d < 2023-01-01 | Selection sample and ledger series |
| Holdout | 2023-01-03 ≤ d < 2025-01-01 | Touched once, after the ledger is written |

Feature history before 2016 may use bars back to the 2014 fetch start.
Those bars are not scored. Holdout returns are not an input to the winner,
the ledger ratio, or PBO.

## Grid

Dip: lookback {10, 20, 40} × hold {5, 10, 21} × decay {0.75, 1.0} = 18
cells. Trend: fast {20, 50, 100} × slow {100, 200} × vol window {20, 60},
dropping fast ≥ slow, = 10 cells. Baseline: 1 cell. Total 29. All 29 are
trials.

## Costs

`configs/backtest.yaml`, which inherits `configs/base.yaml`: commission 1
bp, half-spread 5 bp, impact coefficient 0.1, turnover bp 0, borrow 50 bp
per year, financing 0, frictionless off, participation limit 0.1, fills at
the next open. Risk gate: gross 2, net 0.2, name 0.05, participation 0.1,
order notional 1e6, predicted vol 0.4. Starting cash 1,000,000. These
limits are not opened up so that a book can trade.

## What gets recorded

Every cell goes through `ProvenanceDB.insert_trial` on the validation
net-of-cost simple returns. Parameter cells of one strategy share a
`cluster_id`. The winner is the highest validation per-period mean/std
(ddof 1). Ties break on `trial_id`. A zero-vol validation path is stored
and cannot win.

PBO is CSCV with 16 blocks on train-then-validation returns, holdout left
out, trailing length divisible by 16. The gate's own report does not load
return series, so its PBO field stays empty; the receipt carries the CSCV
number. Deflated-ratio thresholds in `quant_fund.reality.report` are not
edited. The gate deflates by cluster count. The receipt also reports the
same deflated-ratio function with the raw recorded trial count, labeled as
that supplemental figure.
