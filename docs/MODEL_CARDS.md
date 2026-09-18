# Model cards

Each trained artifact must record the fields below (MLflow params/tags plus `docs` dump). Fill per experiment; this file is the template.

## Required fields

- Family (ranking, alpha, distribution, volatility, covariance, regime, tail, liquidity, fusion)
- Model class and version
- Git commit
- Feature set name and version
- Label name and horizon
- Universe version
- Train / validation / test date ranges
- Hyperparameters and Optuna trial count
- Random seed
- Dataset fingerprint
- Cost assumptions
- Statistical metrics (IC, pinball, QLIKE, …)
- Economic metrics (net spread, turnover, drawdown)
- Promotion level reached
- Known limitations
- Data source (`synthetic` vs vendor/file)

Research notebooks additionally emit immutable provenance: run ID, Git
revision, configuration SHA-256, dataset-scope SHA-256, row/column counts,
seed, PIT declaration, and execution claim. A provenance record is required
for institutional review; it does not turn synthetic data into deployable
evidence.

## Families in this repo

| Family | Baseline | ML |
|---|---|---|
| Ranking | composite, ridge, ElasticNet | XGB/LGBM, LambdaRank, XE-NDCG |
| Alpha | historical mean, ridge residual | trees |
| Distribution | empirical, Gaussian, linear QR | XGB/LGBM quantile |
| Volatility | rolling, EWMA, causal GARCH/EGARCH/GJR, HAR-RV | XGB/LGBM |

Rolling and EWMA baseline cards use `vol_20` and `vol_ewma`, respectively,
which are sigma features. Training squares those forecasts before QLIKE so the
forecast and `future_realized_var_*` response are both in variance units;
`vol_of_vol` is never used as a proxy for either baseline.

### GARCH-family card requirements

A GARCH artifact must record the causal fit input (`ret_1` decimal returns), the
forward evaluation label and horizon, `variance_units=decimal_squared`, `arch` percent
scaling, distribution (`normal`, `t`, or `skewt`), convergence status, persistence
validation, observation count, and any fallback reason. Walk-forward evaluation
refits from an expanding history strictly before each test-date origin, so one
fold-boundary forecast is never repeated across future test dates. The return history
comes from the full feature panel, not rows retained by forward-label filtering, so
newly observed `ret_1` values remain available even when their future label is null.
The current pooled panel fit must also record
`series_scope=date_level_equal_weight_cross_section`; this is not an
asset-specific volatility forecast and must not be consumed as one without a
keyed-per-security artifact.
Forecast consumers must use
`forecast(...)["variance"]` or `cumulative_variance`; the legacy `predict()` sigma
adapter is not valid input to variance QLIKE. Current training artifacts are research
outputs and are not automatically connected to live/risk consumers.

Model artifacts are published transactionally: training serializes to a same-directory
temporary file, flushes it, and atomically replaces the final `.joblib` path only after
the write succeeds. An interrupted or failed retrain therefore leaves the previously
published artifact intact. The existing joblib format remains compatible, but artifacts
must still come only from a trusted model directory because `joblib.load()` is
pickle-based and is not a safe deserializer for attacker-controlled files. A
published artifact may also have a sibling `<artifact>.sha256` sidecar. When
present, loading verifies the SHA-256 digest before deserialization and rejects
malformed or mismatched sidecars; a missing sidecar remains supported for legacy
artifacts. The sidecar detects corruption or replacement relative to a trusted
sidecar, but it does **not** authenticate artifact origin and must not be treated
as a substitute for trusted-directory controls, signed provenance, or access
control.

### HAR-RV card requirements

The persisted HAR artifact is a supervised volatility regressor. Training and
inference must consume the same declared five-column volatility feature schema
(`vol_20`, `vol_ewma`, `vol_parkinson`, `ret_1`, `vol_of_vol`); the forward
realized-variance label is response-only and must never be used to reconstruct
features. Non-finite feature rows are excluded consistently at fit time, and the
artifact records the v2 supervised input contract. `har_design(...)` remains an
opt-in helper for callers that explicitly provide a causal historical RV series;
it is not silently applied to forward labels inside `fit()`.

| Covariance | sample, EWMA, LW, factor | DCC |
| Regime | 1-state, vol threshold | Gaussian HMM |
| Tail | historical, Gaussian | distribution-derived, drawdown classifier |
| Liquidity | half-spread, bps, sqrt impact | Almgren–Chriss |
| Northset | OHLC/session identities, SYNTHETIC L2, OHLC vol, Kyle/Roll/OFI/VPIN | none (ADR-007) |
| Conformal | Split CQR + ACI wrapping Gaussian / linear QR / historical tail | blocked until ADR-007 + coverage baseline |

Do not treat feature importance as causal. SHAP is optional and unlabeled as causal.


## Overnight honesty (2026-09-15/16)

All overnight Wave 1–19 benches and paper/research smokes use **SYNTHETIC** data
(or explicitly labeled research fixtures). Metrics are infrastructure /
research-only diagnostics. **`live_pnl_claim=false`** always — do not treat
SYNTHETIC IC, Sharpe-like ratios, paper NAV, or PERF timings as live P&L or
promotion evidence. Vendor market-data and live broker fills remain out of scope.
