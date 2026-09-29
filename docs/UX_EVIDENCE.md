# UX Evidence (ULTRAPLAN P4.5)

Self-check output, `--help` surface coverage, and error-message quality for the
`dipcatcher` CLI (the `quant` entry point is an alias — same app object).

## `dipcatcher doctor` self-check

`doctor` reports environment readiness without ever raising: a missing or
unreadable config is reported as `config: unreadable: <ExcType>` and the rest
of the check still runs on defaults (fail-visible, not fail-crash). Observed on
a clean checkout (`dipcatcher doctor`, trimmed to one line per field):

```
firm: Artificial Hedge
product: Dipcatcher
version: 0.4.0
benchmark_catalog_version: 2
families: ranking,alpha,volatility,distribution,regime,tail,drawdown,
  liquidity,reinforcement,conformal,evalues,jackknife_plus,crc,
  weighted_conformal,interval_risk,quantile_bandit,cv_plus,cpcv,
  localized_conformal,conformal_rank,online_crc,portfolio_conformal,northset
mode: research
live_allowed: False
config: ok
dir_raw..dir_metadata: ok
data_manifest: ok
research_receipt: ok
core_imports: ok
default_fill: next_open
core_kline_engine: robinhood_plus
robinhood_plus_backend: numpy
fx1_package: ok
fx1_moonshot_key: unset
fx1_signing_key: unset
```

Every line is a checkable fact: catalog version, the full benchmark-family
list, runtime mode (research — `live_allowed: False` is the honesty contract in
one line), the five data dirs, manifest + receipt presence, import health, the
default fill policy, and which optional secrets are configured. `REJECTED`
appears only if `mode=live` while `allow_live` is false.

## `--help` coverage

`dipcatcher --help` lists ~40 commands; every one carries a one-line summary
stating its contract class — research / SYNTHETIC / research_only / "no live
broker" / "not a live P&L claim" appear wherever a command could be mistaken
for market evidence:

```
verify-ledger, audit-record, audit-trace, doctor, ingest, collect,
build-features, build-labels, validate, forecast, kronos-forecast, optimize,
backtest, candle-book, kyle-ofi, research, execution-sensitivity,
verify-identities, fleet, verify-receipt, vol-bench, rankic, capacity,
session-book, vendor-book-map, book-panel, northset, verify-research, report,
tearsheet, api, paper, monitor, sim-live, train, hmm, ls, pit, qm,
research100, proof, leakage, reality, proofcore, stress, lineage, lake
```

Per-command `--help` exposes every option with a description and default
(e.g. `vol-bench --help` documents `--models`, `--shards`, `--horizons`,
`--min-history`, `--n-origins`, `--stride`, `--n-bars`, `--seed`,
`--out-dir`, `--receipt-version`).

## Error-message quality

All command-facing validation goes through `typer.BadParameter` — rendered as
`Invalid value: <reason>` in an error box, exit code 2, no traceback. Sampled
failures (2026-09-28):

| Invocation | Result |
|---|---|
| `vol-bench --models bogus_model` | `Invalid value: unknown vol model 'bogus_model'` |
| `rankic --panels bogus` | `Invalid value: unknown panel 'bogus'; have ['linear_signal', 'monotone_cubic', 'pure_noise', 'regime_flip', 'weak_signal']` |
| `capacity --dev --books bogus` | `Invalid value: unknown book generator: bogus` |
| `fleet --shards bogus` | `Invalid value: unknown fleet shard 'bogus'` |
| `validate --bogus-flag` | `No such option: --bogus-flag` |
| `verify-receipt /nonexistent.json` | structured JSON `"errors": ["receipt_unreadable:FileNotFoundError"]`, exit 1 |
| `backtest --config /missing.yaml` | `Invalid value: config file not found: /missing.yaml` (was a raw `FileNotFoundError` traceback before this PR — now translated in `cli/_app._cfg`, covering every config-taking command) |
| `doctor --config /missing.yaml` | `config: unreadable: FileNotFoundError` line, check continues (doctor must report, not crash) |

Named enum violations always list the valid values. Structural verifier
errors come back as machine-readable error lists rather than prose. No bare
`except:` → silent-failure paths were found on the CLI surface; the audit lane
(`reality`, `proof`, `proofcore`, `leakage`) all exit non-zero on violation.

## Real-data spine (Yahoo EOD, 13 names incl. SPY benchmark)

Every read-side command exercised end-to-end on a real collected tape
(2026-09-28, `collect --source yahoo`, 5,148 bars, `source: parquet`):

| Command | Result |
|---|---|
| `collect` | `raw/sources/yahoo.parquet` + sidecar receipt (5,148 rows) |
| `promote-bars` | sealed `bar_promotion.v1` receipt; `verify-receipt` → `valid: true` |
| `ingest` | silver bars + universe + `data_manifest.json` |
| `build-features` / `build-labels` | 5,096 × 199 gold features, labels with `label_end_time_*` |
| `train ranking --model ridge` | sealed `model_artifact.v1`; 4,823 rows × 42 features, 120 IC dates, evidence report |
| `forecast` | per-name alpha + Mondrian-CQR intervals |
| `optimize` | ledoit_wolf_2004_linear weights, causal as-of |
| `backtest` | 390 gold-panel decision dates, 4,861 fills, Kupiec/Christoffersen/Acerbi–Szekely, `research_only`/`live_pnl_claim=false` stamps |
| `execution-sensitivity` | latency×impact grid over the same causal dates |
| `paper --max-steps 5` | full run dir: orders/equity/positions/cash_ledger/broker_state/promotion_dry_run, `PAPER_SIMULATED` labels |
| `validate ridge` | correctly fails closed — `evidence_complete: false`, `research_receipt_valid: false` |
| `research` | full proprietary bench suite: split BH-FDR families, 549 discovery DM tests, conformal coverage (~0.90 on 0.95 target), e-process, jackknife+, northset microstructure |
| `audit-record` + `verify-ledger` | 327-entry hash-chained ledger, signed Merkle root, `unsigned_suffix: 0` |
| `audit-trace` | receipt metric → ledger entry 0 with valid inclusion proof |
| `doctor` | all checks `ok`; `research_receipt: missing` reported, not raised |

Two real defects this run caught and fixed: the `backtest`/`paper`/`POST /backtest`
decision grid was never intersected with the gold panel (#363 — the command
raised on warmup dates), and `build_labels` silently emitted all-null
`future_excess_return_*` when the configured benchmark had no tape rows (#364).
