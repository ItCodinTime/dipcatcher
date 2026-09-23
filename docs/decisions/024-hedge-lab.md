# ADR-024: Artificial Hedge fund lab (economic catalog on a paper book)

## Status

Accepted

## Date

2026-09-19

## Context

Dipcatcher already has two honesty catalogs: research family blobs (no Sharpe /
Sortino / Calmar / pnl / nav key tokens) and paper `analytics_export` (those
ratios allowed, `live_pnl_claim` must stay false). SOTA protocol cards score
IC / pinball / CRPS / DM / RankIC / calibration. They do not size a book.

Artificial Hedge needs a **fund lab**: a causal public-ridge book on a file
tape, next-open fills, modeled costs, then Sharpe / Sortino / Calmar / drawdown
/ PSR / MinTRL with RAM-sized block-bootstrap intervals. That work must not
contaminate research family blobs, must not set `live_pnl_claim=true`, must not
promote SYNTHETIC, and must not move `blend_weight` off 0. Payload under
`D:/dipcatcher` is capped at 100 GiB. A 128 GiB machine may use most of RAM
after a 28 GiB OS/Cursor headroom.

## Decision

1. Add package `quant_fund.hedge_lab`. The economic scoreboard is catalog
   `hedge_lab_analytics` (paper/backtest). A Sharpe-free `research_twin` is
   the only object that may be scanned by `family_blob_forbidden_metrics_absent`.
2. `dipcatcher hedge-lab` trains public-feature ridge, builds a causal
   `optimize_asof` weight panel on a rebalance grid, and `run_backtest`s with
   `skip_intervals=true`. Default tape is `configs/hedge_lab.yaml` (parquet
   session-close file lake). `--fetch-tape` expands the Yahoo universe;
   `--fetch-zoo` may snapshot Kronos-small locally. Neither is a CI path.
3. Resource governor: `assert_disk_budget` fails closed above 100 GiB of lab
   payload (`data/`, `third_party/kronos_weights/`, `artifacts/hedge_lab/`).
   `ram_plan` targets 72% of (available, total − 28 GiB headroom). Block
   bootstrap `n_boot` is sized so paths+wealth fit that workspace. Optional
   `claim_workspace` commits pages after the bootstrap so the working set is
   actually large, then drops the array.
4. GitHub may receive JSON receipts. Tapes, weights, parquet equity, and RAM
   maps stay gitignored. Do not push 100 GiB.
5. `blend_weight` remains 0. SYNTHETIC receipts stamp
   `synthetic_not_promotable`. File-tape Sharpe is a paper-book diagnostic,
   not a champion alias and not live P&L.

## Consequences

- Research notebooks stay IC/CRPS/calibration. The fund lab is the place
  Sharpe/Calmar are legal.
- A high paper Sharpe on vendor-adjusted EOD does not move promotion or
  robinhood+ blend.
- Operators can fill tens of GiB of RAM on bootstrap CIs without intending
  to OOM Windows.
