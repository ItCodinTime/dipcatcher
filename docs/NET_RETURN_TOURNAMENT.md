# Step 2: net-return tournament

The tournament reuses step 1's hashed dataset and declared validation/test
dates. It compares a frozen list of strategies under one shared daily replay.
The initial adapters are equal weight, momentum and reversal, with an optional
long/short ranking book. It does not accept arbitrary pretrained models or
unverified precomputed return series.

## Run

First prepare the source benchmark using `REAL_DATA_BENCHMARK.md`. Then:

```bash
uv run python -m quant_fund.research.net_tournament prepare \
  --benchmark-run data/metadata/real_benchmark/us_wide_v1 \
  --spec configs/net_tournament.json \
  --output data/metadata/net_tournament/us_wide_v1

uv run python -m quant_fund.research.net_tournament run \
  --run data/metadata/net_tournament/us_wide_v1 --phase validation

uv run python -m quant_fund.research.net_tournament run \
  --run data/metadata/net_tournament/us_wide_v1 --phase test
```

The example cost values are declared scenarios, not measured venue estimates.
Both configured impact and double impact are run for every strategy. The
equal-weight benchmark and all candidates start each phase with the same
capital, liquidity, cost model, exposure limits, calendar and terminal-exit
rules. Model-specific positions and trade costs naturally differ.

## Decision and accounting rules

- Trading eligibility uses only the signal close and previous observations.
  It deliberately does not use step 1's future-label-complete scoring rows to
  choose tradable names. A shared lookback, equal to the largest candidate
  lookback or 20 sessions, applies to the entire slate.
- All strategies use the same top `max_names` pool ranked by trailing
  20-session dollar volume. Missing/late historical observations disqualify
  the relevant signal or liquidity estimate. No future universe membership is
  consulted. A globally absent session still requires an external calendar
  audit, as in step 1.
- Signal-close NAV and prices determine requested share quantities. Those
  quantities execute at the next session's open. The execution day's close,
  volume and volatility do not enter that order's sizing or cost estimate.
- Commission and half spread charge traded notional in basis points. Impact
  is `notional * Y * lagged_volatility * sqrt(notional / lagged_ADV)`.
  Participation caps can partially fill an order; requested and unfilled
  quantities remain in its fill record.
- Cash is debited for purchases and all costs, and credited for sales.
  Borrow applies to short market value at the preceding open. Cash credit
  and debit interest use the previous cash balance. All rates accrue using
  elapsed calendar time / 365, including weekends.
- Targets leave the configured buffer below gross/name limits. Execution
  checks costs and opening gaps again. Risk-reducing trades can reduce an
  existing breach; an order cannot increase a breach. Rejections retain their
  requested quantity and reason.
- The final session attempts liquidation under the same participation and
  risk rules. Residual positions remain marked and reported. Incomplete
  liquidation blocks `economic_evidence_gate`.
- A missing mark for a held asset fails that trial. Failed trials remain in
  the result, and the full-slate comparison cannot report a statistical win.

This simulator is isolated from `backtest.engine` and `fast_replay`, whose
causal-liquidity changes remain in PR #28. It neither calls nor implicitly
certifies those execution paths. Its supported surface is the bounded daily
tournament described here.

## Selection and evidence

Validation selects the highest mean net return, breaking ties by name. Test
reads that selection from the matching immutable validation receipt and never
reselects the holdout winner. Every configured candidate remains in both
scenario reports, with its complete daily cash/NAV/cost ledger and fill record.

Candidate-minus-benchmark net returns enter the existing stationary-bootstrap
Reality Check, SPA and Romano-Wolf StepM implementations. The block length,
bootstrap count and seed are frozen before scoring. Tests use the complete
slate; insufficient samples, failed trials or degenerate differentials do not
silently reduce the number of candidates. These are finite-sample research
estimates under resampling assumptions, not guarantees.

`selected_holdout_adjusted_rejection` is the selected candidate's positive
holdout mean differential with StepM adjusted p < 0.05. The separate
`economic_evidence_gate` additionally requires all trials to complete,
all terminal liquidations to complete, and the selected candidate's mean
excess return to remain positive with doubled impact. Neither field changes
the repo's champion alias: `promote` and `live_pnl_claim` are always false.

## Receipts and limitations

`manifest.json` binds the benchmark receipt, entire candidate slate, execution
config, source-code hashes and runtime. Each phase first reserves an exclusive
`*.attempt.json`, then writes `validation.json` or `test.json`. A crash leaves
the attempt visible. Completed/failed attempts cannot be overwritten; creating
a new run must retain disclosure of any previously inspected holdout. Hashes
detect accidental tampering but cannot expose undisclosed external trials.

Results are **simulated net price returns**. Dividends, splits, actual borrow
locates, auction depth and exchange-specific fills are not modeled. Input
opens and closes must represent consistent share units; total-return-adjusted
closes paired with raw opens are not valid execution data. Disclosures from
the source benchmark, including vendor survivorship and a previously examined
holdout, are carried into every result. Equal gross limits also do not imply
equal beta, net exposure, or sector risk across strategies.

The tracked market snapshot can be recovered from a normal checkout and used
with the fixed step-1 protocol. Generated-data tests establish accounting and
timing behavior; any market-snapshot run must publish its complete candidate
slate, both impact scenarios and sealed validation/test receipts. A passing
simulation remains retrospective evidence, subject to the source disclosures
above, and cannot authorize live execution.

The 2026-09-25 frozen snapshot run, full ledgers, and negative economic result
are documented in `data/metadata/research/phase1_20260925.md`. The portable
index at `data/metadata/research/phase1_evidence_index.json` verifies the
receipt chain and links the frozen configuration.
