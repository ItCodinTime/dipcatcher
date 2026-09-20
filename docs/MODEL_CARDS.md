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
| Ranking | composite, ridge, ElasticNet | XGB/LGBM, LambdaRank, XE-NDCG, RFF / ridgeless (Kelly–Malamud–Zhou), SDF ridge / EN (Kozak–Nagel–Santosh), IPCA / IPCA-\(\alpha\) (Kelly–Pruitt–Su), RP-PCA (Lettau–Pelger), FNW group LASSO, Giglio–Xiu 3-pass, FGX double-selection, Fama–MacBeth, GKX PCR/PLS/GBRT, Kelly–Pruitt 3PRF, Kelly–Malamud–Pedersen principal portfolios |
| K-line foundation | robinhood+ NumPy hierarchical BSQ + Markov decoder | optional Kronos torch weights (`[nn]`) |
| Alpha | historical mean, ridge residual | trees |
| Distribution | empirical, Gaussian, linear QR | XGB/LGBM quantile |
| Volatility | rolling, EWMA, causal GARCH/EGARCH/GJR/APARCH/FIGARCH, HAR-RV | XGB/LGBM |

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
asset-specific volatility forecast. Per-security causal forecasts are a
separate `garch_name_forecasts_asof` namespace (`series_scope=security_level_ret_1`)
and must not be consumed as the market overlay.
Forecast consumers must use
`forecast(...)["variance"]` or `cumulative_variance`; the legacy `predict()` sigma
adapter is not valid input to variance QLIKE. `forecast_asof` consumes
`vol_garch.joblib` as a **date-level equal-weight market overlay**: it clones the
artifact specification and refits on `ret_1` dates strictly before the decision
origin, then scales optimizer covariance so equal-weight market variance matches
the one-step GARCH variance. Paper/backtest `check_order` prefers the
Parkinson Realized GARCH overlay for `max_predicted_vol` when
`vol_realized_garch.joblib` is present (Wave 117); otherwise it uses this
return-only one-step market sigma. Per-security `vol_20` remains the
name-level impact/cost field. Walk-forward QLIKE for the pooled GARCH artifact
is the date-level equal-weight realized-variance score on nonoverlapping
origins (stride = label horizon); overlapping-date QLIKE is a diagnostic only.
The volatility bench uses the same date-level collapse and Hansen–Hodrick
lags of at least \(h-1\). Per-security walk-forward QLIKE/density lives in
`garch_name_walk_forward` (`scoring_scope=security_level_ret_1`, Wave 114)
and must not be consumed as the market overlay score. Realized GARCH
(`RealizedGARCHVol`, Wave 115) is a separate Parkinson-daily-OHLC namespace
and does not replace this overlay, `vol_20`, or covariance scaling.
Paper/backtest `check_order` prefers the Parkinson Realized GARCH one-step
sigma for `max_predicted_vol` when `vol_realized_garch.joblib` is present
(Wave 117); otherwise it keeps this return-only overlay. This is a
research-lab risk overlay, not a live or high-frequency RV claim.

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

| Covariance | sample, EWMA, LW 2004, OAS, NL LW 2020, factor | one-step Gaussian / Student-t DCC, scalar CES ADCC, diagonal/full AG-DCC, and Bollerslev CCC \(H_{t+1}\); named RiskMetrics EWMA one-step path; named trailing OAS / analytical nonlinear LW / unbiased sample plus overlay; named CCC one-step path |
| Regime | 1-state, vol threshold | Gaussian HMM |
| Tail | historical, Gaussian | distribution-derived, drawdown classifier |
| Liquidity | half-spread, bps, sqrt impact | Almgren–Chriss |
| Northset | OHLC/session identities, SYNTHETIC L2, OHLC vol, Kyle/Roll/OFI/VPIN | none (ADR-007) |
| Conformal | Split CQR + ACI wrapping Gaussian / linear QR / historical tail | still wraps baselines; robinhood+ paths may feed raw quantiles |

### robinhood+ card requirements

A robinhood+ artifact must record `family=kline_foundation`, `backend`
(`numpy` or `torch`), `decoder`, `price_space=split_adjusted`, lookback,
`pred_len`, `sample_count`, `s1_bits` / `s2_bits`, and Kronos attribution
(arXiv:2508.02739, MIT). The NumPy path is a hierarchical BSQ tokenizer plus
s1-then-s2 autoregression — not a claim that Hub weights were loaded. Torch
weights require the `[nn]` extra and local checkpoints unless
`allow_network=true`. Default `blend_weight` is 0 until robinhood+ beats
**public-feature** ridge (oracle columns dropped) on a **non-synthetic** PIT
tape, Diebold–Mariano does not prefer ridge on CRPS, and the calibration
gate is green (Jackknife+ ≥ 1−2α, CQR/ACI Kupiec recorded, PIT KS recorded).
SYNTHETIC IC is an engine-correctness diagnostic and cannot take a champion
alias. Forecasts enter fusion; they do not bypass the risk gate.
`execution_claim=research_only`. The name is internal; not affiliated with
Robinhood Markets, Inc.

Do not treat feature importance as causal. SHAP is optional and unlabeled as causal.


## Overnight honesty (2026-09-15/16)

All overnight Wave 1–19 benches and paper/research smokes use **SYNTHETIC** data
(or explicitly labeled research fixtures). Metrics are infrastructure /
research-only diagnostics. **`live_pnl_claim=false`** always — do not treat
SYNTHETIC IC, Sharpe-like ratios, paper NAV, or PERF timings as live P&L or
promotion evidence. Vendor market-data and live broker fills remain out of scope.
