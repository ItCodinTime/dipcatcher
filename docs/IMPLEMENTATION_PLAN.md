# Implementation plan

Phased build of `quant_fund`. After each phase: tests, ruff, mypy, docs, limitations. Do not skip to neural networks.

## Phase 1 — Project / config / testing

uv, Python 3.12, typed config, schemas, logging, seeds, CLI `dipcatcher doctor`, CI.

## Phase 2 — Point-in-time data

Protocols, parquet/CSV + synthetic adapters, security master, PIT universe, corporate actions, leakage tests.

## Phase 3 — Features and labels

OHLCV families, per-timestamp CS transforms, horizon labels, `FeatureMetadata`.

## Phase 4 — Validation

Walk-forward, purge, embargo, CPCV, DSR/PSR, Optuna restricted to train.

## Phase 5 — Backtester and costs

Event-driven engine, next-bar fills, spread/impact/borrow, metrics. Frictionless runs labeled.

## Phase 6 — Ranking and alpha

Ridge / ElasticNet / composite rank; XGB/LGBM; LambdaRank with date groups.

## Phase 7 — Return distribution

Empirical, Gaussian, linear QR, tree quantiles, pinball/CRPS/crossing.

## Phase 8 — Volatility

Rolling/EWMA baselines plus a causal GARCH-family subsystem (GARCH, EGARCH,
GJR, APARCH, FIGARCH; Gaussian, Student-t, skew-t), explicit return/variance
units, convergence and stationarity gates, horizon-indexed probabilistic
forecasts, PIT diagnostics, and variance-contract QLIKE. Walk-forward
evaluation must use historical `ret_1` as the fit input and forward realized
variance only as the evaluation label. Walk-forward GARCH QLIKE is date-level
and overlap-aware (nonoverlapping *h*-step origins). Walk-forward also scores
the one-step date-level `ret_1` density (log-score, CRPS, PIT). Per-security
causal GARCH is a separate `security_level_ret_1` as-of namespace and does not
replace the date-level overlay. Realized-GARCH is implemented as a separate
log-linear Gaussian namespace whose realized measure is one-day Parkinson
variance from daily OHLC (not high-frequency RV). Paper/backtest
`check_order` uses that Parkinson overlay for `max_predicted_vol` when the
artifact is present; `forecast_asof` / `optimize_asof` consume the same overlay
for market variance and covariance scaling. Name-level `vol_20` remains the
impact/cost sigma. Intraday realized measures remain
unavailable; this phase makes no SOTA or live-performance claim.

## Phase 9 — Covariance

Sample, EWMA, Ledoit–Wolf, OAS, factor, DCC, PSD repair.

## Phase 10 — Portfolio optimizer

CVXPY mean-variance + TC + CVaR mode, constraint diagnostics, risk contributions.

## Phase 11 — Regime

Gaussian HMM probabilities, BIC/AIC/OOS likelihood, no Sharpe-driven \(k\).

## Phase 12 — Tail risk

VaR/ES, drawdown probabilities, stylized stress.

## Phase 13 — Liquidity / impact / execution

Square-root impact, Almgren–Chriss, TWAP/VWAP baselines, misspecification tests.

## Phase 14 — Fusion

Configurable transparent score, ablations, and an opt-in cross-fitted ridge
stacker for research diagnostics. Production remains on transparent fusion.

## Phase 15 — Registry

MLflow aliases, fingerprints, promotion gates (levels 1–8).

## Phase 16 — API, reporting, monitoring

FastAPI, HTML/Markdown reports, drift, kill switch.

## Phase 17 — Paper / shadow

Live clock with simulated fills; shadow challengers; live gated.

**Status (overnight Waves 1–32, 2026-09-15→16):** **DONE for sim paper/shadow** —
`SimulatedBroker`, `dipcatcher paper` CLI, ReplayClock/WallClock, ledger under
`data/metadata/paper/` (schema v2 + `validate_ledger_schema`), kill+risk on every
order, shadow / multi-challenger no-capital, resume/persist, promotion dry-run
(`would_promote_live` always false; `live_pnl_claim=false`), analytics_export.json
+ `validate_analytics_export`, 100+/200-step resume stress tests.
API honesty stamp (Wave 31): responses force `research_only=true` /
`live_pnl_claim=false` (overwrite poisoned upstream). Research scorecard rejects
Sharpe/Sortino/Calmar/pnl/nav family keys (Wave 32).
**Still missing:** vendor market-data adapter, live broker / real fills
(intentionally SKIP tonight — not faked).

## Phase 18 — Profiling

Only after correctness. Polars/DuckDB first.

**Status (overnight Waves 1–32, 2026-09-15→16):** Panel + ranker + conformal +
wrappee caches landed; event-time day-index + order-preserving `history_prefix_upto`
path for causal calibration (Wave 12 recovered causal 25d ≈ **3.19s** vs Wave 11
≈4.33s; Wave 5 best ≈2.84s — not fully restored). Wave 28 re-bench in
`docs/PERF.md` + `data/metadata/perf_bench.json` (wave:28). Dominant remaining
cost is still conformal Student-t MLE per asof. **Parallel causal dates blocked**
by sequential `w_prev` (SKIP — correctness > wall). Batched polars asof filters
deferred.

## Outstanding limitations (living)

- No vendor market-data adapter (CSV/Parquet + synthetic only) — SKIP overnight.
- No live broker adapter / real fills — SKIP overnight.
- Parallel causal dates (`w_prev` sequential) — SKIP (unsafe to parallelize).
- Causal 25d wall still ~0.29–0.35s above Wave 5 best on this machine after prefix path.
- Fundamentals / options / macro features require PIT release timestamps; not faked.
- Neural nets remain behind extra `[nn]` except robinhood+ (ADR-023): the
  Kronos-derived K-line engine is a core forecast path with a NumPy default;
  official Kronos weights are optional and offline-by-default.
- Nonlinear covariance shrinkage deferred.
- Torch GPU determinism is not claimed.
- Cross-fitted ridge stacking is research-only: chronological folds, explicit
  warm-up rows, finite-data checks, and leakage rejection are required. It is
  not enabled in production fusion and cannot authorize promotion.
- First-run demo uses SYNTHETIC data. Negative or noisy IC is reported honestly.
- Paper/backtest/API metrics are `research_only` / `live_pnl_claim=false` — never claim live P&L.
- Research family blobs must not carry Sharpe/Sortino/Calmar/pnl/nav keys (scorecard + verify fail closed).

## Overnight honesty stamps (Waves 25–32)

- **CPCV per-group purge** (Wave 25): purge+embargo applied per test group (union-span bug fixed); integrity smoke Wave 26.
- **TrialLedger empty DSR** (Wave 27): empty ledger → honest NaN DSR (no invented score).
- **PERF Wave 28**: re-bench + `docs/PERF.md` / `perf_bench.json` updated; parallel causal still SKIP.
- **API honesty stamp** (Wave 31): FastAPI backtest/portfolio/risk/forecast/ranking paths force `research_only=true` / `live_pnl_claim=false`.
- **Forbidden-metrics regression** (Wave 32): catalog helper + scorecard/verify reject Sharpe/Sortino/Calmar/pnl/nav keys in family blobs.
- **Test count**: ~832+ non-network (`pytest -m 'not network'`); no Cursor; no commits; no fake live Sharpe.
