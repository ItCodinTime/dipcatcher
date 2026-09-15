# Implementation plan

Phased build of `quant_fund`. After each phase: tests, ruff, mypy, docs, limitations. Do not skip to neural networks.

## Phase 1 — Project / config / testing

uv, Python 3.12, typed config, schemas, logging, seeds, CLI `quant doctor`, CI.

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

Rolling, EWMA, GARCH, HAR-RV, tree models, QLIKE.

## Phase 9 — Covariance

Sample, EWMA, Ledoit–Wolf, factor, DCC, PSD repair.

## Phase 10 — Portfolio optimizer

CVXPY mean-variance + TC + CVaR mode, constraint diagnostics, risk contributions.

## Phase 11 — Regime

Gaussian HMM probabilities, BIC/AIC/OOS likelihood, no Sharpe-driven \(k\).

## Phase 12 — Tail risk

VaR/ES, drawdown probabilities, stylized stress.

## Phase 13 — Liquidity / impact / execution

Square-root impact, Almgren–Chriss, TWAP/VWAP baselines, misspecification tests.

## Phase 14 — Fusion

Configurable transparent score, ablations, optional OOF stacking later.

## Phase 15 — Registry

MLflow aliases, fingerprints, promotion gates (levels 1–8).

## Phase 16 — API, reporting, monitoring

FastAPI, HTML/Markdown reports, drift, kill switch.

## Phase 17 — Paper / shadow

Live clock with simulated fills; shadow challengers; live gated.

## Phase 18 — Profiling

Only after correctness. Polars/DuckDB first.

## Outstanding limitations (living)

- No vendor market-data adapter (CSV/Parquet + synthetic only).
- No live broker adapter.
- Fundamentals / options / macro features require PIT release timestamps; not faked.
- Neural nets behind extra `[nn]`, unimplemented until baselines exist.
- Nonlinear covariance shrinkage deferred.
- Torch GPU determinism is not claimed.
- Learned fusion stacking is not enabled; transparent fusion only.
- First-run demo uses SYNTHETIC data. Negative or noisy IC is reported honestly.
