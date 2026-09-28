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
