# Research references

Methodological anchors. Implementations may differ; deviations are in `MATH_SPEC.md`.

## Cross-sectional asset pricing / ML

- Gu, Kelly, Xiu (2020). Empirical Asset Pricing via Machine Learning. *Review of Financial Studies*.
- Learning-to-rank (LambdaRank, XE-NDCG) as used in LightGBM.
- Factor-neutral portfolio construction (standard industry practice; Grinold–Kahn).

## Distributional forecasting

- Koenker, Bassett (1978). Regression Quantiles.
- Pedersen, Thomas Q. Predictable Return Distributions.
- Gneiting, Raftery (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. *JASA*. CRPS, pinball.

## Volatility

- Bollerslev (1986). GARCH.
- RiskMetrics EWMA.
- Corsi (2009). HAR-RV.
- Patton (2011). Volatility forecast comparison / QLIKE.

## Covariance

- Ledoit, Wolf (2004). A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices.
- Engle (2002). Dynamic Conditional Correlation.
- Engle, Sheppard (2001). Theoretical and empirical properties of DCC.
- Higham (1988). Computing a nearest symmetric positive semidefinite matrix.

## Regime

- Hamilton (1989). A new approach to the economic analysis of nonstationary time series.
- Rabiner (1989). A tutorial on hidden Markov models.

## Tail risk

- Acerbi, Tasche (2002). On the coherence of expected shortfall.
- Rockafellar, Uryasev (2000). Optimization of conditional value-at-risk.
- Optional EVT / peaks-over-threshold (Embrechts et al.) — not in v1 core.

## Execution

- Almgren, Chriss (2000). Optimal execution of portfolio transactions.
- Implementation shortfall (Perold). Square-root impact as a baseline, not a law of nature.

## Backtest robustness

- Bailey, López de Prado. The Deflated Sharpe Ratio; Probabilistic Sharpe Ratio.
- Bailey, Borwein, López de Prado, Zhu. Probability of Backtest Overfitting / CSCV.
- López de Prado. *Advances in Financial Machine Learning* — purged/embargoed CV, CPCV.
- Diebold, Mariano (1995). Comparing predictive accuracy.

## Conformal prediction

- Romano, Patterson, Candès (2019). Conformalized Quantile Regression. *NeurIPS*.
- Gibbs, Candès (2021). Adaptive Conformal Inference Under Distribution Shift. *NeurIPS*.
- Vovk, Gammerman, Shafer. *Algorithmic Learning in a Random World* — Mondrian conformal predictors.
- Angelopoulos, Bates. A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.

## Portfolio

- Markowitz mean-variance.
- Grinold, Kahn. *Active Portfolio Management* — IC, ICIR, residual alpha.
