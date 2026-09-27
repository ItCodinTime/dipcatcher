# Audit ledger and observability

This page describes two optional lab facilities:

- `quant_fund.audit` is an append-only ledger for research receipts and simulated paper events.
- `quant_fund.observe` is tracing, metrics, and JSON logs. It is off unless `DIPCATCHER_OBSERVE` is set.

Neither facility submits orders, talks to a broker, or changes sealed research receipts. Simulation gauges are operational telemetry. They are not research scores and they are not a live P&L claim.

## Ledger

A ledger is a directory:

| File | Role |
|---|---|
| `entries.jsonl` | One canonical JSON object per entry, append-only |
| `checkpoints.jsonl` | Signed Merkle tree heads, append-only |
| `ed25519.pub` | Convenience copy of the signing public key. Not a trust anchor |
| `.lock` | `flock` so two writers do not interleave lines |

Entry kinds are `research_run`, `paper_decision`, `simulated_order`, `fill`, and `risk_decision`. Each entry stores `prev_hash` and `entry_hash = SHA-256(canonical body)`. The body omits `entry_hash`. The genesis `prev_hash` is 64 zero hex digits. Canonical JSON is sorted keys, tight separators, ASCII-escaped, with no NaN tokens. A line that is not exactly those bytes is `noncanonical_line`.

`research_run` payloads reject top-level keys in `FORBIDDEN_RESEARCH_METRIC_KEYS` (`sharpe`, `sortino`, `calmar`, `pnl`, `nav`). `live_pnl_claim` is recorded as `false`.

### Merkle tree

The tree is RFC 6962, in log order. Leaf preimage is the canonical entry body. Leaf hash is `SHA-256(0x00 || body)` and node hash is `SHA-256(0x01 || left || right)`. The empty log is `SHA-256()`. Inclusion and consistency proofs follow RFC 6962 sections 2.1.1 and 2.1.2. Verification uses the iterative Certificate Transparency audit path and rejects a proof that is shorter or longer than the tree size requires.

`quant_fund.proofcore` hashes a **sorted** set of leaves. That construction cannot detect reordering, so the audit ledger does not call it.

### Signatures

Every `sign_every` entries (default 1) the ledger appends a checkpoint. The signature covers only `v`, `scheme`, `key_id`, `tree_size`, `merkle_root`, and `timestamp_utc`.

Ed25519 uses the `cryptography` package already installed with the harness. The private key is a mode-0600 hex file and never appears in a ledger line. Signatures are deterministic.

Sigstore is optional and fail-closed. `sign_with_sigstore` imports `sigstore` and reads `DIPCATCHER_SIGSTORE_ID_TOKEN`. A missing package, a missing token, or a token that is not a usable OIDC token raises `SignatureUnavailableError` and does not write a placeholder bundle. CI does not contact Fulcio or Rekor. Set `DIPCATCHER_SIGSTORE_INSTANCE` to `production` or `staging` when you sign for real. Verification of a Sigstore checkpoint requires the package and a pinned identity (`--sigstore-identity`). An unpinned identity is an error.

### `verify-ledger`

```bash
uv run verify-ledger LEDGER --trust-pub key.pub --expect-size N --expect-root HEX
uv run dipcatcher verify-ledger LEDGER
```

Exit 0 means the bytes that are present have an intact hash chain and every checkpoint signature matches the Merkle root of that prefix. Exit 1 is an inconsistent or tampered log. Exit 2 is usage, including an unreadable `--trust-pub`.

`valid` can be true while `fully_signed` is false when a signed prefix is intact and a later suffix has not been checkpointed yet. A non-empty log with no checkpoints is `unsigned_log` and is not valid.

Two truncations are internally consistent and need an outside witness:

- Deleting the last entry **and** its checkpoint leaves a shorter log that still verifies. `--expect-size` and `--expect-root` are that witness.
- Replacing the log and its signatures with a new Ed25519 key verifies against the bundled public key. `--trust-pub` detects it. `ed25519.pub` beside the log is attacker-controlled if the directory is attacker-controlled.

### Linking a published number

`dipcatcher audit-record --receipt` appends a `research_run` whose `receipt_sha256` is `quant_fund.research.verify._receipt_digest`, the same digest `verify-research` uses. The entry copies `run_id`, `git_revision`, `git_worktree_sha256`, `config_sha256`, `dataset_sha256`, `dataset_content_sha256`, `northset_inputs_sha256`, and `code_sha256` from the receipt. The receipt file is not rewritten.

`dipcatcher audit-trace --metric dotted.path` checks that the ledger verifies, the digest matches an entry, those provenance fields match the receipt, the path exists, and an inclusion proof rebuilds the Merkle root. `checkout_git_revision` is the current `git rev-parse HEAD`. Comparing the worktree hash is off by default because that hash walks the tree. `--verify-receipt` embeds the existing research verifier and does not by itself decide `linked`.

`dipcatcher audit-record --paper-dir` reads `orders.parquet`, hashes the file and each row, and appends `paper_decision`, `simulated_order`, `fill` (when the fill price and quantity are finite), and `risk_decision`. Equity parquet is hashed as bytes only. The parquet files are not rewritten. This is a record of a simulation, not an order route.

Recording is explicit. The simulated broker, risk gate, ingest, feature, and forecast modules do not import this package.

## Observability

Set `DIPCATCHER_OBSERVE` to `1`, `true`, `yes`, or `on`. Otherwise `span()` does not allocate ids, import an OpenTelemetry SDK, record metrics, write logs, or open a socket. `import quant_fund.observe` does not import `opentelemetry` or `prometheus_client`. The harness speaks a small OTLP/HTTP JSON subset to `DIPCATCHER_OTEL_ENDPOINT` (for example `http://127.0.0.1:4318/v1/traces`) and renders Prometheus text 0.0.4 from memory.

Stages are fixed: `ingest`, `features`, `model`, `decision`, `simulated_execution`. When the flag is on, the CLI wraps those five callables. The wrapper calls the original function and re-raises its exception. `uninstall_passive_hooks` restores the originals. With the flag off, the CLI does not wrap them and does not import those modules for instrumentation.

| Metric | Kind | Meaning |
|---|---|---|
| `dipcatcher_stage_latency_seconds` | histogram | Stage latency. Buckets are upper bounds, not interpolated quantiles |
| `dipcatcher_errors_total` | counter | Stage errors, plus `otel_export` when a span post fails |
| `dipcatcher_data_freshness_seconds` | gauge | Age of the newest input a caller reports |
| `dipcatcher_simulation_pnl` | gauge | Simulated mark-to-market. Not a research headline |
| `dipcatcher_simulation_gross_exposure` | gauge | Simulated gross exposure |
| `dipcatcher_simulation_net_exposure` | gauge | Simulated net exposure |

`DIPCATCHER_METRICS_PORT` serves `/metrics` on `DIPCATCHER_METRICS_HOST` (default `127.0.0.1`). JSON logs carry `correlation_id`. Field names ending in `_key`, `_token`, or `_secret`, and the names `password`, `key`, `token`, `secret`, and `authorization`, are redacted. `key_id` is not a secret.

SLO definitions live in `deploy/observability/slo.yml`. A missing sample is `no_data`, not a pass. The evaluator does not claim that a research or live process meets the objectives.

### Local stack

```bash
docker compose -f deploy/observability/docker-compose.yml up
```

The compose file publishes the collector, Prometheus, and Grafana on loopback only. Prometheus scrapes `host.docker.internal:9464`, so the process must set `DIPCATCHER_METRICS_HOST=0.0.0.0` inside the host namespace Docker can reach (or run the process where that address is reachable). Grafana provisions datasource uid `prometheus` and the dashboard `deploy/observability/grafana/dashboards/dipcatcher.json`. Anonymous access is a local viewer. No admin password is committed.

```bash
export DIPCATCHER_OBSERVE=1
export DIPCATCHER_OTEL_ENDPOINT=http://127.0.0.1:4318/v1/traces
export DIPCATCHER_METRICS_PORT=9464
export DIPCATCHER_METRICS_HOST=0.0.0.0
```

### Overhead

`quant_fund.observe.overhead.measure` times an empty call, a disabled span, and an enabled span on one process. The OTLP endpoint is cleared for that run, so the numbers exclude network export. They are not a benchmark and they are not a research result. Run it in the environment you care about; do not copy a number from another machine into a claim about this one.

```bash
uv run python -c "import json; from quant_fund.observe.overhead import measure; print(json.dumps(measure(), indent=2))"
```

## Tests and CI

`make audit-obs` runs `tests/unit/audit`, `tests/unit/observe`, and mypy on the two packages. The `audit-observability` GitHub job does the same and checks that the compose file parses. Tamper tests edit, drop, reorder, rehash, re-sign, and truncate ledger bytes. Hook tests assert that the production modules do not reference these packages and that a wrapped simulated submit matches a direct submit.

`make test` still collects these tests because they live under `tests/unit`.
