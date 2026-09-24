# Institutional readiness matrix

Dipcatcher is Artificial Hedge's research lab. This matrix is deliberately
evidence-gated: a capability is not marked ready because a module exists or a
synthetic fixture passes.

| Control area | Current state | Evidence / gate |
|---|---|---|
| Point-in-time data contract | Implemented for file adapters | PIT timestamp validation, duplicate-key checks, OHLCV checks, manifest hash verification, and decision-time filtering for late cross-sectional/market aggregates |
| Configuration safety | Implemented | Unsupported data sources, non-finite simulator parameters, and non-stationary synthetic persistence fail validation |
| Data lineage | Implemented | `dipcatcher doctor` requires source identity, four canonical artifacts, SHA-256s, nonnegative row counts, and data-root containment |
| Research reproducibility | Implemented | Immutable receipt binds Git revision, dirty-worktree hash, config, dataset content, runtime, package versions, and benchmark catalog version |
| SOTA scientific benches | Implemented as research diagnostics on the lab panel | Canonical verifier requires the versioned 23-family catalog; fixture-only toys are `dgp=fixture` and excluded from panel Kupiec H-rows; panel-or-skip (empty blob) fails scorecard honesty rather than silently mixing DGPs |
| Validation integrity | Implemented | Walk-forward, purge/embargo, CPCV audit, HAC/DM, multiple-testing and conformal family gates |
| Numerical risk controls | Implemented | Covariance inputs are finite and square; PSD repair is deterministic eigenvalue clipping with finite-output guarantees |
| Execution research | Implemented as simulation | Next-open fills, costs, participation, Almgren–Chriss/TWAP comparisons, and tamper-evident backtest artifacts |
| Paper / shadow operation | Implemented as simulation | Crash-resumable simulated broker; resume appends prior orders/equity/shadow-equity/positions/cash rows and fill receipts; deterministic target/exposure ordering; kill switch, champion/shadow dry-run, no live capital |
| Promotion | Fail-closed | Synthetic evidence, missing metrics, non-finite metrics, and invalid receipts cannot promote |
| CI reproducibility | Implemented | Frozen `uv.lock`, lock consistency check, full tests, type/lint checks, canonical receipt verification |
| CI token scope | Implemented | Workflow `GITHUB_TOKEN` is restricted to repository contents read access; no job in CI requires write access |
| Package distribution | Implemented | CI builds wheel and source distribution, installs the wheel without the checkout installed, and runs the installed `dipcatcher --help` entry point |
| API security boundary | Implemented for local/service operation | API-key authentication, loopback fail-closed default, constant-time key comparison, secure response headers, artifact-root containment, 64 KiB streamed request-body cap, and strict request schemas |
| Dependency and code security | Implemented as CI gates | Locked dependency `pip-audit` report, medium/high Bandit gate, and retained machine-readable audit artifact |
| Container operations | CI-gated | Non-root runtime user, loopback default, healthcheck, immutable dependency sync, host-local optional MLflow binding; CI builds the image and waits for its healthcheck |
| Vendor market data | Not implemented | Only synthetic and local CSV/Parquet adapters are available; no vendor entitlement is implied |
| Live broker connectivity | Not implemented | No live orders, broker credentials, or live P&L claims are supported |
| Live readiness | Blocked by missing external evidence | Requires authorized vendor data, broker adapter, operational controls, and independently verified holdout/forward evidence |


## Dual honesty catalogs (research vs analytics export)

Two different surfaces — do not conflate:

1. **Research scorecard / family blobs** — `FORBIDDEN_RESEARCH_METRIC_KEYS` /
   `family_blob_forbidden_metrics_absent` reject sharpe/sortino/calmar/pnl/nav
   *key tokens* in lab headlines.
2. **Paper/backtest `analytics_export`** — may nest equity `nav_*` and stress
   `*_pnl` diagnostics under `ANALYTICS_SCHEMA_KEYS`; honesty gate is
   `live_pnl_claim=false` via `validate_analytics_export` (fail-closed on true).

Never treat a valid analytics export as a research family scorecard blob.
`/ready` is fail-closed when `data_manifest` or the immutable research receipt is
missing/invalid (the configs allowlist resolves `_CONFIGS_DIR` before containment
checks). A green readiness response therefore proves both data provenance and
research-artifact integrity for the configured research runtime.

## Minimum evidence before any live claim

All of the following must be present and independently reviewable:

1. A licensed, point-in-time data source with release and ingestion timestamps.
2. A broker adapter with authenticated order and fill reconciliation.
3. A non-synthetic holdout and deployment-shaped forward/shadow record.
4. Cost, liquidity, borrow, financing, and failure-mode measurements from the
   target venue.
5. A signed promotion receipt whose immutable verifier passes and whose live
   authorization is explicit.

Until those conditions exist, Dipcatcher outputs are research, backtest, or
simulated paper/shadow evidence only.
