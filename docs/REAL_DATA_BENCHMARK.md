# Step 1: fixed real-data forecast benchmark

This workflow makes data assumptions, eligible observations, and split
boundaries explicit before publishing any model score. It complements the
existing scientific benches and promotion gates. It always reports
`promote=false` and `live_pnl_claim=false`.

## Run

From an installed checkout with the tracked dataset present:

```bash
uv run python -m quant_fund.research.real_benchmark prepare \
  --protocol configs/real_benchmark_us_wide.json \
  --output data/metadata/real_benchmark/us_wide_v1

uv run python -m quant_fund.research.real_benchmark score \
  --run data/metadata/real_benchmark/us_wide_v1 --phase validation

uv run python -m quant_fund.research.real_benchmark score \
  --run data/metadata/real_benchmark/us_wide_v1 --phase test
```

The example is a **retrospective diagnostic** of the existing vendor pool.
Its holdout is explicitly marked previously inspected. The committed
`data/file_us_wide/metadata/v2_lane5_pit_universe.json` already reports results
on 2025 and discloses vendor survivorship. A new protocol cannot erase that
history. The example's adjustment assumptions and timestamp provenance also
remain unverified. It is a starting contract, not a validated dataset or a
production acceptance report. Its fixed split dates were not selected from
new benchmark scores.

The tracked Parquet snapshot is available through a normal Git checkout. Its
SHA-256 must match the frozen protocol (`e22bf3eff634c742299b6495f8daf8f02adcc6cda47d9641e3094c0c191ae117`)
before any scores are reported. Git LFS gold features and labels are not inputs
to this workflow. If preparation rejects timestamps or coverage, investigate
the source; do not change the hash or timing rules merely to pass the gate.

## Data and timing contract

- Input is a daily Parquet panel with `security_id`, `event_time`,
  `available_time`, `ingested_time`, `source`, and the declared price column.
- All timestamps must be timezone-aware. Securities share one timestamp per
  UTC session date; multiple sessions within a day are unsupported.
- `event_time` denotes the completed observation; the decision cutoff is
  `event_time + decision_delay_seconds`. A late observation excludes every
  sample window that uses it. Missing sessions are never bridged as if they
  were one-session returns. Completely absent market-wide sessions cannot be
  discovered without an external exchange calendar; the observed calendar is
  used and must be checked against the intended market before deployment.
- Nulls, nonpositive/nonfinite prices, duplicate revisions, invalid timestamps,
  insufficient samples, and synthetic/fixture source labels fail validation.
- The source URL, usage basis, price adjustments, universe selection,
  availability reconstruction, and previous holdout inspection are mandatory
  declarations. Their truth cannot be established by a file hash.
- Features use the current and prior 20 sessions only. Labels must end within
  their split; training/validation label endpoints are additionally separated
  from the next split by the configured number of observed sessions.

## Matched baselines

Every baseline is scored on identical eligible rows. Metrics first average
errors across securities on each date, then average across dates. This keeps
dates with a larger universe from silently receiving more weight.

| Baseline | Prediction |
|---|---|
| Zero | Zero return |
| Historical mean | Pooled mean target from the fixed training split |
| Rolling mean 20 | Mean of the last 20 one-session returns, scaled by horizon |
| Ridge | Fixed-alpha regression using return 1/5/20 and volatility 20 |

Ridge normalization, coefficients, intercept and the historical mean are fit
only on the training split. No auto-selection or test-driven refit occurs.
The rolling mean is a deliberately simple forecast, not a compounded-return
identity. MSE and MAE are forecast diagnostics; transaction costs, financing,
execution prices, and portfolio returns are outside this first milestone.

## Receipts and remaining work

`prepare` writes an exclusive `manifest.json` with the frozen protocol,
dataset hash, code hash, runtime versions, eligibility counts and limitations.
It validates the whole panel for integrity but emits no test performance.
`score` checks the same data/code/runtime, requires the matching validation
receipt before the test phase, and refuses to overwrite either phase.
The sealed manifest records a dataset path relative to its run directory so a
complete checkout (including both `data/file_us_wide/` and the run directory)
can be moved without rewriting the receipt. Move the entire directory layout
together; a moved run alone cannot resolve its dataset. The tournament likewise
records the relative location and exact sealed contents of this benchmark run.

These controls prevent accidental reuse within a run directory. A user can
copy data, inspect it elsewhere or recreate a run, so this is not cryptographic
proof of an untouched holdout. A genuinely fresh forward period requires an
externally timestamped protocol and collection after that timestamp.

Next milestones: (2) matched net-return tournament and complete trial record;
(3) cost-aware allocation; (4) forward shadow ledger; (5) measured performance
and operational improvements. Step 2 should reuse this dataset/split contract
and first resolve or explicitly retain its data limitations.
