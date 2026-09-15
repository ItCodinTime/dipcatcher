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

## Families in this repo

| Family | Baseline | ML |
|---|---|---|
| Ranking | composite, ridge, ElasticNet | XGB/LGBM, LambdaRank, XE-NDCG |
| Alpha | historical mean, ridge residual | trees |
| Distribution | empirical, Gaussian, linear QR | XGB/LGBM quantile |
| Volatility | rolling, EWMA, GARCH, HAR-RV | XGB/LGBM |
| Covariance | sample, EWMA, LW, factor | DCC |
| Regime | 1-state, vol threshold | Gaussian HMM |
| Tail | historical, Gaussian | distribution-derived, drawdown classifier |
| Liquidity | half-spread, bps, sqrt impact | Almgren–Chriss |
| Conformal | Split CQR + ACI wrapping Gaussian / linear QR / historical tail | blocked until ADR-007 + coverage baseline |

Do not treat feature importance as causal. SHAP is optional and unlabeled as causal.
