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

## Foundation-model comparison targets

- Shi et al. (2025). Kronos: A Foundation Model for the Language of Financial
  Markets. *NeurIPS 2025* (arXiv:2508.02739); repo `shiyu-coder/Kronos` (weights under HF org `NeoQuasar`). Used as a
  published SOTA comparison target for causal one-step-ahead return-distribution
  forecasting; see `scripts/sota_eval_kronos.py` and `.dsh-24x7/evidence-sota-eval-*.json`.
- Ansari et al. (2024/25). Chronos / Chronos-Bolt / Chronos-2: Amazon's
  time-series foundation models (`amazon-science/chronos-forecasting`, weights
  under HF org `autogluon`). Chronos-2 evaluated via native quantile output
  (quantile-integral CRPS) under the same causal protocol.
- Das et al. (2024). TimesFM: Google's decoder-only time-series foundation
  model (`google/timesfm`, weights `google/timesfm-2.5-200m-pytorch`), evaluated
  zero-shot under the same protocol.

## Data-snooping / multiple testing

- White, H. (2000). A Reality Check for Data Snooping. *Econometrica* 68(5).
- Hansen, P. R. (2005). A Test for Superior Predictive Ability. *Journal of Business & Economic Statistics* 23(4).
- Romano, J. P., Wolf, M. (2005). Stepwise Multiple Testing as Formalized Data Snooping. *Econometrica* 73(4).
- Hansen, P. R., Lunde, A., Nason, J. M. (2011). The Model Confidence Set. *Econometrica* 79(2).
- Politis, D. N., Romano, J. P. (1994). The Stationary Bootstrap. *JASA* 89(428).
- Politis, D. N., White, H. (2004). Automatic Block-Length Selection for the Dependent Bootstrap. *Econometric Reviews* 23(1).

## Conformal prediction

- Romano, Patterson, Candès (2019). Conformalized Quantile Regression. *NeurIPS*.
- Gibbs, Candès (2021). Adaptive Conformal Inference Under Distribution Shift. *NeurIPS*.
- Vovk, Gammerman, Shafer. *Algorithmic Learning in a Random World* — Mondrian conformal predictors.
- Angelopoulos, Bates. A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.

## Portfolio

- Markowitz mean-variance.
- Grinold, Kahn. *Active Portfolio Management* — IC, ICIR, residual alpha.

---

# Research references (Day Wave 1 update — 2026-09-16)

Concrete citations for SOTA methods used or targeted by this repo. Map each to `src/quant_fund/` (or tests).
**Honesty:** SYNTHETIC benches are research-only; never treat as live P&L / vendor fills.

## Conformal prediction (finance / time series, 2023–2026)

| Citation | Why it matters | Repo map |
|----------|----------------|----------|
| Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*, arXiv:2107.07511 (updated guide) | Split CP, coverage semantics, nonconformity | `models/conformal*.py`, CRC paths, fixtures in `tests/unit/test_*conformal*` |
| Gibbs & Candès, Adaptive Conformal Inference (ACI) / DtACI line | Non-exchangeable sequential coverage | Online CRC / adaptive λ paths (`online_crc`, CRC extremes) |
| Angelopoulos et al., Conformal Risk Control (CRC) | Control E[loss] not just miscoverage | `crc` modules + `test_crc.py` |
| Kaya et al., *Conformal Prediction for Reliable Stock Selections*, PMLR 2025 | CP sets for stock selection | Ranking / conformal rank benches |
| *Conformal Predictive Portfolio Selection (CPPS)*, arXiv:2410.16333 | Portfolio-level conformal intervals | `portfolio_conformal` + extremes tests |
| Cocouvi (2025), *CP for Financial Time Series Under Asset Pricing Models* | ICP / EnbPI / Mondrian / time-weighted vs GARCH | Weighted / localized conformal |
| QDtACI VaR conformal layer, *Risk Management* (2026) | Model-agnostic intervals around VaR | Risk backtest + conformal calibration |
| Adaptive conformal for VaR/ES on crypto, *J. Risk Financial Manag.* 17(6) 2024 | ACI vs GARCH for market risk | Risk battery (Kupiec/Christoffersen/ES) |
| Barber, Candès, Ramdas & Tibshirani, *Predictive Inference with the Jackknife+*, Ann. Statist. 49(1) (2021) | Marginal exchangeable coverage ≥1−2α (Jackknife+/CV+); agg-specific CV+ floors | `models/jackknife_plus.py`, `models/cv_plus.py`, `bench_jackknife_plus` / `bench_cv_plus`; H10/H15 bound floors |
| Bian & Barber, *Training-conditional coverage for distribution-free predictive inference*, Electron. J. Statist. 17 (2023) 2044–2066 (arXiv:2205.03647); follow-ons on stability | Guarantees are typically **marginal**; training-conditional coverage can fail for Jackknife+ without stability — do **not** read H10/H15 `coverage_floor` as training-conditional | Day Wave 41 `coverage_guarantee_scope=marginal_exchangeable` on JP/CV+ meta + benches |

## CPCV / PBO (López de Prado / Bailey)

| Citation | Why it matters | Repo map |
|----------|----------------|----------|
| López de Prado, *Advances in Financial Machine Learning* (2018), ch. 7 & 12 | Combinatorial purged CV, purge + embargo | CPCV implementation + per-group purge (Wave 25+) |
| Bailey, Borwein, López de Prado & Zhu, *The Probability of Backtest Overfitting*, J. Computational Finance 20(4) (2017); PDF: davidhbailey.com/dhbpapers/backtest-prob.pdf | PBO via CSCV; PBO>0.5 stop sign | PBO / TrialLedger / DSR paths |
| Bailey & López de Prado, Deflated Sharpe / PSR | Multiple-testing Sharpe adjustment | PSR/DSR fixtures (Wave 8) |
| Bailey & López de Prado, MinTRL (minimum track-record length) | Samples needed for PSR ≥ conf | `min_track_record_length` (Day Wave 2); `min_trl_from_returns` → `book_diagnostics["min_trl"]` smoke (Day Wave 5) |

## Proper scoring rules (Gneiting)

| Citation | Why it matters | Repo map |
|----------|----------------|----------|
| Gneiting & Raftery, *Strictly Proper Scoring Rules, Prediction, and Estimation*, JASA 102(477) (2007) | CRPS, energy score, propriety | `crps_from_quantiles` (Riemann); `crps_gaussian` / `mean_crps_gaussian` (closed-form); `crps_student_t` / `mean_crps_student_t` (Day Wave 45); `crps_empirical` (ensemble); bench keys `crps_gaussian_closed` / `dm_crps_*` (Day Wave 8) + `crps_scaled_student_t_closed` (Day Wave 45); MATH_SPEC |
| Jordan, Krüger & Lerch, *Evaluating Probabilistic Forecasts with scoringRules*, J. Stat. Softw. 90(12) (2019) | Closed-form CRPS for parametric families incl. location-scale Student-t | `crps_student_t` / `mean_crps_student_t`; bench `crps_scaled_student_t_closed` (Day Wave 45); MATH_SPEC |
| Gneiting et al., *Probabilistic Forecasts, Calibration and Sharpness* | PIT, sharpness subject to calibration | PIT diagnostics |
| Pinball / quantile loss (Koenker; Gneiting quantile scores) | Proper for τ-quantile | Pinball / mean_pinball |
| Annual Review: *Proper Scoring Rules for Estimation and Forecast Evaluation* (2024/25) | Modern loss-orientation convention | Distribution extremes tests |

## Diebold–Mariano & sequential comparison

| Citation | Why it matters | Repo map |
|----------|----------------|----------|
| Diebold & Mariano, *Comparing Predictive Accuracy*, JBES (1995/2002) | Loss-differential test for **forecasts** | DM pairwise helpers |
| Diebold, *Comparing Predictive Accuracy, Twenty Years Later* (NBER WP 18391) | DM is for forecasts not models | Research agent / scorecard honesty |
| Choe & Ramdas, *Comparing Sequential Forecasters*, Operations Research 72(4) (2023) | Anytime-valid CS / e-processes for score diffs | `e_process_loss_diff` / `e_process_dm` (Day Wave 4); optional pairwise DM flag; `bench_volatility` `e_dm_*` (Day Wave 9); `bench_distribution` `e_dm_crps_*` (Day Wave 18) + scaled `e_dm_crps_scaled_*` (Day Wave 19) |

## E-values / testing by betting

| Citation | Why it matters | Repo map |
|----------|----------------|----------|
| Grünwald, de Heide, Koolen; Ramdas et al. e-value / Ville inequality line | Anytime-valid evidence; capital process | `evalues` + Ville fixtures |
| Choe & Ramdas (2023) above | E-processes for forecaster comparison | `metrics.evalues` loss-diff capital + Ville; `pairwise_diebold_mariano(..., include_e_process=True)`; vol bench `e_dm_*`; distribution bench `e_dm_crps_*` (Day Wave 18) + `e_dm_crps_scaled_*` (Day Wave 19) |

## Portfolio conformal / risk / execution

| Citation | Why it matters | Repo map |
|----------|----------------|----------|
| CPPS arXiv:2410.16333 | Conformal portfolio selection | `portfolio_conformal` |
| Almgren & Chriss, *Optimal Execution of Portfolio Transactions*, J. Risk 3 (2001) | Mean–variance execution frontier, IS | `almgren_chriss.py` + extremes (Wave 22) |
| Kyle, *Continuous Auctions and Insider Trading*, Econometrica 53 (1985) | λ from signed volume vs mid change | Northset `kyle_lambda` |
| Roll, *A Simple Implicit Measure of the Effective Bid-Ask Spread*, JF 39 (1984) | \(2\sqrt{-\gamma_1}\) from mid-change autocov | Northset `roll_spread` |
| Parkinson, *The Extreme Value Method for Estimating the Variance of the Rate of Return*, JB 53 (1980) | Range variance vs close-to-close | Northset `parkinson_vs_close_to_close` |
| Garman & Klass, *On the Estimation of Security Price Volatilities from Historical Data*, JB 53 (1980) | OHLC variance | Northset `garman_klass_vs_close_to_close` |
| Rogers & Satchell, *Estimating Variance from High, Low and Closing Prices*, Annals of Applied Probability (1991) | Drift-robust OHLC variance | Northset `rogers_satchell_vs_close_to_close` |
| Yang & Zhang, *Drift-Independent Volatility Estimation…*, JB 73 (2000) | Overnight + RS combination | Northset `yang_zhang_variance` |
| Corwin & Schultz, *A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices*, JF 67 (2012) | High-low spread | Northset `corwin_schultz_spread` |
| Amihud, *Illiquidity and stock returns*, JFM 5 (2002) | \|r\| / dollar volume | Northset `amihud_illiquidity` |
| Cont, Kukanov & Stoikov, *The Price Impact of Order Book Events*, J. Financial Econometrics (2014) | OFI | Northset `order_flow_imbalance` |
| Abdi & Ranaldo, *A Simple Estimation of Bid-Ask Spreads from Daily Close, High, and Low Prices*, RFS 30 (2017) | Close-high-low spread | Northset `abdi_ranaldo_spread` |
| Barndorff-Nielsen & Shephard, *Econometrics of Testing for Jumps in Financial Economics…*, JFE 4 (2006) | Bipower vs RV jump share | Northset `session_bipower_jump` |
| Easley, López de Prado & O’Hara, *Flow Toxicity and Liquidity…*, RFS 25 (2012) | VPIN (bulk-volume proxy) | Northset `vpin_proxy` / `session_vpin` |
| Osler, *Currency Orders and Exchange Rate Dynamics: An Explanation for the Predictive Success of Technical Analysis*, JF 58 (2003) | Stop-cluster / sweep dynamics | Northset `liquidity_sweep_frame` |
| Acerbi & Tasche, Expected Shortfall coherence / ES definition | ES as coherent risk measure | ES metrics + Acerbi–Székely style backtests |
| Kupiec POF; Christoffersen (1998) independence/CC | VaR hit-rate & clustering tests | `kupiec_pof` / `christoffersen_*` + `var_backtest_hooks`; `bench_tail` ind/CC (Day Wave 16) |
| Acerbi & Székely (2014), Backtesting Expected Shortfall | Unconditional Z1 / conditional Z2 ES tests | `acerbi_szekely_z1`/`z2` + `var_backtest_hooks` (Day Wave 2); `bench_tail` primary + unscaled/scaled (Day Wave 15) |
| Fissler & Ziegel (2016), Higher order elicitability and Osband's principle | Joint (VaR, ES) elicitability | `fissler_ziegel_loss` / `mean_fissler_ziegel` (Day Wave 3); `bench_tail.fissler_ziegel_mean` (Day Wave 15) |
| Nolde & Ziegel (2017), Elicitability and backtesting | FZ0 0-homogeneous joint score; comparative backtests | FZ0 formula in MATH_SPEC + `var_backtest_hooks.fissler_ziegel_mean` + `bench_tail` (Day Wave 15) |

## Repo honesty anchors

- `docs/MATH_SPEC.md`, `docs/VALIDATION.md`, `docs/DATA_SOURCE_LABELS.md`
- `FORBIDDEN_RESEARCH_METRIC_KEYS` / `family_blob_forbidden_metrics_absent` (no Sharpe/Sortino/Calmar/pnl/nav in **research family** headlines; paper `analytics_export` may nest equity/stress pnl/nav under `live_pnl_claim=false`)
- Paper/validate: `would_promote_live=false` on SYNTHETIC; `live_pnl_claim=false`
