# Research references

Methodological anchors. Implementations may differ; deviations are in `MATH_SPEC.md`.

## Cross-sectional asset pricing / ML

- Fama, MacBeth (1973). Risk, Return, and Equilibrium: Empirical Tests. *Journal of Political Economy*. Date-level CS slopes averaged; catalog `fm`. Ridge per-date slopes: `fm_ridge`.
- Jegadeesh (1990). Evidence of Predictable Behavior of Security Returns. *Journal of Finance*. Short-term reversal; `classic` sign on `cs_z_reversal_1`.
- Jegadeesh, Titman (1993). Returns to Buying Winners and Selling Losers. *Journal of Finance*. Skip-week 20-session momentum; `cs_z_mom_skip_5_20`.
- Blitz, Huij, Martens (2011). Residual Momentum. *Journal of Empirical Finance*. CAPM residual 20-session momentum; `cs_z_idio_mom_20`.
- Bali, Cakici, Whitelaw (2011). Maxing Out: Stocks as Lotteries. *Journal of Financial Economics*. MAX; `classic` short `cs_z_max_ret_20`.
- Ang, Hodrick, Xing, Zhang (2006). The Cross-Section of Volatility and Expected Returns. *Journal of Finance*. Idiosyncratic vol; `classic` short `cs_z_idio_vol_60`.
- Amihud (2002). Illiquidity and Stock Returns. *Journal of Financial Markets*. `classic` long `cs_z_amihud`.
- George, Hwang (2004). The 52-Week High and Momentum Investing. *Journal of Finance*. `classic` long `cs_z_high_52w_prox`.
- Gu, Kelly, Xiu (2020). Empirical Asset Pricing via Machine Learning. *Review of Financial Studies*. NBER w25398. PCR `pcr`, PLS `pls`, Huber GBRT `gbrt`. Neural nets not implemented (ADR-007). Linear autoencoder ≡ IPCA.
- Kelly, Pruitt (2015). The Three-Pass Regression Filter: A New Approach to Forecasting Using Many Predictors. *Journal of Econometrics* 186(2). Automatic-proxy 3PRF Tables 1–2; catalog `tprf`. PLS is the no-intercept special case.
- Kelly, Malamud, Pedersen (2023). Principal Portfolios. *Journal of Finance* 78(1). Open access DOI [10.1111/jofi.13199](https://doi.org/10.1111/jofi.13199). NBER w27388. \(\Pi=E[R_{t+1}S_t']\); catalog `pp`.
- Kelly, Malamud, Zhou (2024). The Virtue of Complexity in Return Prediction. *Journal of Finance* 79(1). Open access DOI [10.1111/jofi.13298](https://doi.org/10.1111/jofi.13298). NBER w30217. RFF (Rahimi–Recht) + ridge \(\hat\beta(z)=(zI+T^{-1}S'S)^{-1}T^{-1}S'R\); catalog `rff`. Adaptations in MATH_SPEC.
- Kozak, Nagel, Santosh (2020). Shrinking the Cross-Section. *Journal of Financial Economics*. NBER w24070. SDF ridge \(b=(\Sigma+zI)^{-1}\mu\) (`sdf_ridge`) and HJ-distance elastic net eq. 28 (`sdf_en`).
- Kelly, Pruitt, Su (2019). Characteristics Are Covariances: A Unified Model of Risk and Return. *Journal of Financial Economics*. NBER w24540. IPCA ALS, \(\Gamma'\Gamma=I_K\); catalog `ipca`. Joint unrestricted \(F_{\mathrm{aug},t}=(1,f_t)'\): `ipca_alpha`.
- Rapach, Strauss, Zhou (2010). Out-of-Sample Equity Premium Prediction: Combination Forecasts and Links to the Real Economy. *Review of Financial Studies*. Equal-weight univariate OLS; catalog `combo`. Non-negative train date-IC weights: `combo_ic`.
- Zou (2006). The Adaptive Lasso and Its Oracle Properties. *Journal of the American Statistical Association*. Catalog `alasso`.
- Lettau, Pelger (2020). Factors That Fit the Time Series and Cross-Section of Stock Returns. *Review of Financial Studies* 33(5). NBER w24858. RP-PCA \(S=(1/T)X'X+\gamma\bar X\bar X'\); catalog `rp_pca`.
- Giglio, Xiu (2021). Asset Pricing with Omitted Factors. *Journal of Political Economy*. NBER w23527. Three-pass PCA / CS / TS estimator; catalog `gx3pass`.
- Freyberger, Neuhierl, Weber (2020). Dissecting Characteristics Nonparametrically. *Review of Financial Studies* 33(5). NBER w23227. Adaptive group LASSO on quadratic splines of rank-transformed characteristics; catalog `fnw`.
- Feng, Giglio, Xiu (2020). Taming the Factor Zoo: A Test of New Factors. *Journal of Finance*. Post-double-selection LASSO then OLS; catalog `ds_lasso`.
- Learning-to-rank (LambdaRank, XE-NDCG) as used in LightGBM.
- Factor-neutral portfolio construction (standard industry practice; Grinold–Kahn).

## Distributional forecasting

- Koenker, Bassett (1978). Regression Quantiles.
- Pedersen, Thomas Q. Predictable Return Distributions.
- Gneiting, Raftery (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. *JASA*. CRPS, pinball.

## Volatility

- Bollerslev (1986). GARCH.
- RiskMetrics EWMA.
- Parkinson (1980). The Extreme Value Method for Estimating the Variance of the Rate of Return.
- Hansen, Huang, Shek (2012). Realized GARCH: A Joint Model for Returns and Realized Measures of Volatility. *Journal of Applied Econometrics*.
- Corsi (2009). HAR-RV.
- Patton (2011). Volatility forecast comparison / QLIKE.

## Covariance

- Ledoit, Wolf (2004). A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices. Default `optimizer.covariance=ledoit_wolf` is trailing 2004 linear shrinkage plus the GARCH/RGARCH overlay (Wave 139) and stays Ledoit–Wolf when \(T\le N\), not silent sample and not one-step \(H_{t+1}\). Named `optimizer.covariance=sample` is trailing unbiased (`ddof=1`) sample covariance plus the overlay (Wave 132). Factor stays unwired.
- Ledoit, Wolf (2020). Analytical Nonlinear Shrinkage of Large-Dimensional Covariance Matrices. *Annals of Statistics*. Named `optimizer.covariance=ledoit_wolf_nonlinear` is trailing analytical 2020 spectral shrinkage plus the overlay (Wave 140), not 2004 linear shrinkage and not numerical QuEST (2017). Generic `ledoit_wolf_2017` / `quest` stay unknown.
- Chen, Wiesel, Eldar, Hero (2010). Shrinkage Algorithms for MMSE Covariance Estimation. *IEEE Transactions on Signal Processing*. Named `optimizer.covariance=oas` is trailing listwise OAS plus the GARCH/RGARCH overlay (Wave 131), not Ledoit–Wolf and not one-step \(H_{t+1}\).
- J.P. Morgan (1996). RiskMetrics Technical Document. Named `optimizer.covariance=ewma` is one-step \(H_{t+1}\) on the trailing complete window (Wave 130), not listwise-deleted in-sample last \(H_t\).
- Engle (2002). Dynamic Conditional Correlation.
- Bollerslev (1990). Modelling the Coherence in Short-run Nominal Exchange Rates: A Multivariate Generalized ARCH Model. *Review of Economics and Statistics*. Catalog `ccc` is two-stage Gaussian GARCH + constant \(R\) (Wave 133). Named `optimizer.covariance=ccc` is one-step \(H_{t+1}\) without overlay (Wave 134). Not Engle DCC.
- Engle, Sheppard (2001). Theoretical and empirical properties of DCC.
- Cappiello, Engle, Sheppard (2006). Asymmetric Dynamics in the Correlations of Global Equity and Bond Returns. Scalar `adcc` is a catalog estimator (Wave 128) and a named optimizer path (Wave 129). Diagonal `agdcc` is a catalog estimator (Wave 135) and a named optimizer path (Wave 136). Unrestricted full-matrix `agdcc_full` is a catalog estimator (Wave 137) and a named optimizer path (Wave 138). Student-t DCC is a catalog estimator (Wave 126) and a named optimizer path (Wave 127).
- Bauwens, Laurent, Rombouts (2006). Multivariate GARCH models: a survey. Covariance Student-t DCC likelihood (scale \(((ν-2)/ν)R\)).
- Higham (1988). Computing a nearest symmetric positive semidefinite matrix.

## Regime

- Hamilton (1989). A new approach to the economic analysis of nonstationary time series.
- Rabiner (1989). A tutorial on hidden Markov models.
- Jurafsky, Martin. *Speech and Language Processing* 3ed, Appendix A
  (Hidden Markov Models). https://web.stanford.edu/~jurafsky/slp3/A.pdf
  Discrete first-order \(\lambda=(A,B,\pi)\); Eisner ice-cream HMM.
  Implemented as `quant_fund.hmm` (Wave 155). Does not replace
  ADR-006 `GaussianHMMRegime`.

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
| Gneiting & Raftery, *Strictly Proper Scoring Rules, Prediction, and Estimation*, JASA 102(477) (2007) | CRPS, logarithmic score, energy score, propriety | `crps_from_quantiles` (Riemann); `crps_gaussian` / `mean_crps_gaussian` (closed-form); `crps_student_t` / `mean_crps_student_t` (Day Wave 45); `crps_empirical` (ensemble); `log_score_gaussian` / GARCH one-step `log_score_one_step` (Day Wave 105); bench keys `crps_gaussian_closed` / `dm_crps_*` (Day Wave 8) + `crps_scaled_student_t_closed` (Day Wave 45); MATH_SPEC |
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

## Option pricing / vol / allocation (quant-models)

- Black, Scholes (1973); Merton (1973). BSM with dividend yield. `quant_models.black_scholes`, `greeks`.
- Cox, Ross, Rubinstein (1979). Binomial tree, American early exercise. `quant_models.binomial`.
- Heston (1993); Albrecher, Mayer, Schoutens, Tistaert (2007) little-trap CF. `quant_models.heston`.
- Gatheral, Jacquier (2014). Arbitrage-free SVI. `quant_models.svi`.
- Nelson, Siegel (1987); Svensson (1994). NSS zeros. `quant_models.nss`.
- López de Prado (2016). Building Diversified Portfolios that Outperform Out of Sample. *Journal of Portfolio Management*. HRP. `quant_models.hrp`.
- Raffinot (2018). Hierarchical Clustering-Based Asset Allocation. HCAA. `quant_models.hrp.hcaa_weights`.
- Maillard, Roncalli, Teiletche (2010). The Properties of Equally Weighted Risk Contribution Portfolios. ERC. `quant_models.risk_parity`.
- Moskowitz, Ooi, Pedersen (2012). Time Series Momentum. *Journal of Financial Economics*. `quant_models.tsmom`.
- Baltussen, Da, Lammers, Martens (2021). Hedging Demand and Market Intraday Momentum. *Journal of Financial Economics*. GEX last-hour follow/fade. `quant_models.gex.last_hour_decide` (no broker).
- SqueezeMetrics / dealer-gamma sign convention (calls +, puts −; an assumption). `quant_models.gex`.
- Krauss, Do, Huck (2017). Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500. *European Journal of Operational Research*. Linear window in `quant_models.krauss`; trees already `gbrt`. DNN behind ADR-007.
- Gu, Kelly, Xiu (2020). OOS \(R^2\) vs zero forecast. `quant_models.gkx.r2_oos`.
- Blume (1975). Betas and Their Regression Tendencies. *Journal of Finance*. `quant_models.beta.blume_beta`.
- Buehler, Gonon, Teichmann, Wood (2019). Deep hedging. Baseline is discrete BS delta in `quant_models.monte_carlo`; nets behind ADR-007.
- Source notebooks: https://github.com/davidalmeida90/quant-models and README sibling repos. Dipcatcher port: ADR-033.

## Lightspeed / time-series momentum books

- Jegadeesh, Titman (1993). Returns to Buying Winners and Selling Losers. *Journal of Finance*. 63-day total-return score in `lightspeed.momentum`.
- Moskowitz, Ooi, Pedersen (2012). Time Series Momentum. *Journal of Financial Economics*. TQQQ book is a 20/180 EMA rotation, not the 12-month TSMOM sign; TSMOM itself is `quant_models.tsmom`.
- López de Prado (2018). *Advances in Financial Machine Learning*. Meta-labeling + purged CV. `lightspeed.metalabel` is reduce-only.
- Frozen specs: `tqqq-long-full-v1`, `stock-momentum-v1`, `nautica-momentum-v1` from https://github.com/cosmic-hydra/lightspeed. Dipcatcher port: ADR-034. Yahoo expected-metrics there are not Dipcatcher live claims.

## Risk-controlled size (paper book)

- Moreira, Muir (2017). Volatility-Managed Portfolios. *Journal of Finance*. `risk.gates.vol_target`.
- Barroso, Santa-Clara (2015). Momentum has its moments. *Journal of Financial Economics*. Time-varying vol scale on a momentum book; implemented as the same vol-target gate, not a new CS ranker.
- Thorp (2006). The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market. Fractional Kelly `κ μ/σ²`; negative μ → 0 (ADR-032). `risk.gates.kelly_leverage`; `BookRiskOverlay`.
- MacLean, Thorp, Ziemba (2011). *The Kelly Capital Growth Criterion*. Fractional Kelly cap.
- Angelopoulos, Bates, Malik, Jordan (2022). Conformal Risk Control. CRC size on trailing losses. `risk.gates.crc_leverage`. ADR-012 / ADR-036.
- Romano, Wolf (2005). Stepwise Multiple Testing as Formalized Data Snooping. Expanding-window allow/deny size. `risk.gates.stepm_leverage`.
- Asness, Moskowitz, Pedersen (2013). Value and Momentum Everywhere. *Journal of Finance*. Public-OHLCV proxy `vme` (12–1 + George–Hwang 52w; no B/M).
- Moskowitz, Ooi, Pedersen (2012). Time Series Momentum. CS analog `tsmom` on `cs_z_mom_12_1` (the TS book remains `quant_models.tsmom`).
- Krauss, Do, Huck (2017). Linear logistic CS challenger `krauss` on the same purged walk-forward as ridge; DNN behind ADR-007.

## K-line foundation models

- Shi, Fu, Chen, Zhao, Xu, Zhang, Li (2025). Kronos: A Foundation Model for the Language of Financial Markets. *arXiv:2508.02739*. Hierarchical discrete K-line tokens + autoregressive decoder. Dipcatcher engine: **robinhood+** (ADR-023). MIT. https://github.com/shiyu-coder/Kronos
- Zhao, Goyal, Mentzer, Van Gool (2024). Binary Spherical Quantization. *arXiv:2406.07548*. Tokenizer stage used by Kronos / robinhood+.

## Repo honesty anchors

- `docs/MATH_SPEC.md`, `docs/VALIDATION.md`, `docs/DATA_SOURCE_LABELS.md`
- `FORBIDDEN_RESEARCH_METRIC_KEYS` / `family_blob_forbidden_metrics_absent` (no Sharpe/Sortino/Calmar/pnl/nav in **research family** headlines; paper `analytics_export` may nest equity/stress pnl/nav under `live_pnl_claim=false`)
- Paper/validate: `would_promote_live=false` on SYNTHETIC; `live_pnl_claim=false`


## Research canon wave (metrics/models/portfolio/execution)

Serial dependence and unit roots — `quant_fund.metrics.serial`:

- Lo, MacKinlay (1988). Stock Market Prices Do Not Follow Random Walks.
  *Review of Financial Studies* 1. Variance-ratio test (VR(q), z1/z2).
- Chow, Denning (1993). A Simple Multiple Variance Ratio Test. *Journal of
  Econometrics* 58. `chow_denning_test`.
- Wright (2000). Alternative Variance-Ratio Tests Using Ranks and Signs.
  *JBES* 18. `wright_variance_ratio`.
- Ljung, Box (1978); Box, Pierce (1970); Engle (1982) ARCH-LM; Jarque, Bera
  (1980); Bartels (1982) rank test; Wald–Wolfowitz runs test.
- Dickey, Fuller (1979); Kwiatkowski et al. (1992) KPSS; Phillips, Perron
  (1988); Zivot, Andrews (1992) — `unit_root_battery` via `arch.unitroot`.

Extreme value theory — `quant_fund.metrics.extremes`:

- Hill (1975); Pickands (1975); Dekkers, Einmahl, de Haan (1989) tail-index
  estimators.
- Balkema, de Haan (1974) / Pickands (1975) GPD peaks-over-threshold;
  Jenkinson (1955) GEV block maxima; return levels.
- Leadbetter et al. (1983) extremogram; Davis, Mikosch (2009) empirical
  extremogram.

Entropy and fractal diagnostics — `quant_fund.metrics.entropy`,
`quant_fund.metrics.fractal`:

- Shannon (1948); Pincus (1991) ApEn; Richman, Moorman (2000) SampEn;
  Bandt, Pompe (2002) permutation entropy; Kaspar, Schuster (1987)
  Lempel–Ziv; Schreiber (2000) transfer entropy.
- Hurst (1951) R/S; Peng et al. (1994) DFA; Katz (1988); Higuchi (1988)
  fractal dimensions; Sevcik (1998).

Multiple-testing corrections — `quant_fund.validation.fdr`:

- Benjamini, Hochberg (1995); Benjamini, Yekutieli (2001); Holm (1979);
  Hochberg (1988); Bonferroni; Sidak (1967); Storey (2002) pi0.

HAC inference — `quant_fund.metrics.hac`:

- Newey, West (1987); Andrews (1991) kernels and automatic bandwidth;
  Andrews, Monahan (1992) AR(1) prewhitening; Kiefer, Vogelsang (2005).

Fractional differentiation and AFML labeling — `quant_fund.models.fracdiff`,
`quant_fund.labels.barriers`, `quant_fund.features.bars`:

- Lopez de Prado (2018). *Advances in Financial Machine Learning*, ch. 3–5:
  fracdiff weights/FFD/min-d ADF scan; cUSUM events, triple barrier,
  meta-labels, trend scanning, uniqueness/decay weights, sequential
  bootstrap; tick/volume/dollar/imbalance/run bars.
- Hosking (1981). Fractional differencing. *Biometrika* 68.
- Easley, Lopez de Prado, O'Hara (2012) — information-driven bars.

State space and point processes — `quant_fund.models.state_space`,
`quant_fund.models.point_process`, `quant_fund.models.changepoint`:

- Kalman (1960); local level/trend; Elliott, van der Hoek, Malcolm (2005)
  pairs-trading hedge ratio via Kalman.
- Ornstein–Uhlenbeck MLE (exact AR(1) transition); half-life.
- Hawkes (1971). Spectra of some self-exciting point processes — exp-kernel
  MLE, branching ratio, compensator residuals (Ogata 1988), thinning
  simulation (Lewis, Shedler 1979).
- Adams, MacKay (2007) BOCPD; Page (1954) CUSUM; Page–Hinkley; Scott,
  Knott (1974) binary segmentation; Jackson et al. (2005) optimal
  partitioning; Killick, Fearnhead, Eckley (2012) PELT cost function.

Random matrix theory — `quant_fund.models.rmt`:

- Marchenko, Pastur (1967); Laloux et al. (1999); Plerou et al. (2002);
  eigenvalue clipping + detoning (Lopez de Prado 2018, ch. 2);
  Kritzman, Li, Page, Rigobon (2011) absorption ratio; Meucci (2009)
  effective rank; inverse participation ratio.

Portfolio allocators — `quant_fund.portfolio.allocators`:

- Lopez de Prado (2016) HRP; Maillard, Roncalli, Teiletche (2010) ERC;
  Choueifaty, Coignard (2008) maximum diversification; Black, Litterman
  (1992); Kelly (1956)/Thorp (2006); Rockafellar, Uryasev (2000) CVaR LP;
  Hallerbach (2012) volatility targeting.

Execution and impact — `quant_fund.execution.impact`:

- Almgren et al. (2005) permanent/sqrt impact; Bouchaud et al. (2009)
  propagator; Gatheral (2010) no-dynamic-arbitrage; Perold (1988)
  implementation shortfall; Kissell, Glantz (2003) POV/benchmark slippage.

Forecast combinations and bandits — `quant_fund.models.ensemble`,
`quant_fund.models.bandits`:

- Bates, Granger (1969); Granger, Ramanathan (1984); Stock, Watson (2004)
  trimmed/median combinations.
- Auer, Cesa-Bianchi, Fischer (2002) UCB1; Auer et al. (2002) EXP3;
  Thompson (1933); Kaufmann, Cappe, Garivier (2012) Bayes-UCB; Sutton,
  Barto (2018) epsilon-greedy.

Copulas — `quant_fund.models.copula`:

- Sklar (1959); Nelsen (2006); Genest, Favre (2007) IFM; Embrechts,
  McNeil, Straumann (2002); Demarta, McNeil (2005) t copula; Marshall,
  Olkin (1988) Archimedean simulation via frailty.

Parametric risk and drawdown — `quant_fund.metrics.risk_parametric`,
`quant_fund.metrics.drawdown`:

- Cornish, Fisher (1937); Zangari (1996) modified VaR; McNeil, Frey (2000)
  t ES; Checkalov, Uryasev, Zabarankin (2004) CDaR; Keating, Shadwick
  (2002) Omega; Kaplan, Knowles (2004) Kappa; Martin, McCann (1989) ulcer
  index; Zephyr pain/tail ratios; Magdon-Ismail, Atiya et al. (2004)
  expected max drawdown.

Technical-analysis canon — `quant_fund.features.indicators`:

- Wilder (1978) RSI/ATR/DMI/ADX; Appel MACD; Bollinger bands; Keltner;
  Kaufman KAMA; Ehlers Fisher transform; Elder force/Elder-ray; Pring KST;
  Granville OBV; Chaikin ADL/CMF/oscillator/volatility; Arms TRIN; Lane
  stochastics; Williams %R/UO; Lambert CCI; Donchian; Hosoda Ichimoku;
  Chande Aroon/CMO; Botes–Siepman Vortex; Mulloy DEMA/TEMA; Hull HMA;
  MFI/EOM/VWAP/stochRSI/TRIX/zlema.

Cycle and spectral filters — `quant_fund.features.cycles`:

- Goertzel (1958); FFT dominant cycle; Marple (1999) analytic signal /
  Hilbert instantaneous frequency; Ehlers (2013) SuperSmoother, roofing
  filter, bandpass.

Regression and IV diagnostics - quant_fund.metrics.regression:

- Gauss-Markov OLS; White (1980) HC0-HC3; Cribari-Neto (2004) HC4; Newey,
  West (1987) HAC; Durbin-Watson (1950); Breusch-Godfrey (1978);
  Breusch-Pagan (1979); White (1980) het test; Ramsey (1969) RESET;
  Goldfeld-Quandt (1965); Brown-Durbin-Evans (1975) CUSUM/CUSUMSQ; Cook
  (1977) distance; Belsley-Kuh-Welsch (1980) VIF; Theil-Sen; Huber (1964)
  IRLS; Koenker-Bassett (1978) quantile regression; Sargan (1958)/Hansen
  (1982) J test; 2SLS.

Cross-sectional factor models - quant_fund.models.factor_models:

- Fama, MacBeth (1973) two-pass; Shanken (1992) EIV; Fama, French (1993)
  2x3 sorts; Jegadeesh, Titman (1993) momentum; Carhart (1997); Bai, Ng
  (2002) IC criteria; Stock, Watson (2002) diffusion indices; Jensen
  (1968) alpha.

Trend/cycle decomposition - quant_fund.models.filters:

- Hodrick, Prescott (1997); Ravn, Uhlig (2002) lambda rule; Baxter, King
  (1999); Christiano, Fitzgerald (2003); Hamilton (2018); Beveridge,
  Nelson (1981); Corbae, Ouliaris (2006) DFT bandpass.

Realized measures - quant_fund.models.realized:

- ABDL (2001) RV; Barndorff-Nielsen, Hansen, Lunde, Shephard (2008)
  realized kernels; Zhang, Mykland, Ait-Sahalia (2005) TSRV; Jacod et al.
  (2009) pre-averaging; Barndorff-Nielsen, Shephard (2004/2006) bipower;
  Lee, Mykland (2008) jump test; Huang, Tauchen (2005) relative jump;
  Podolskij, Vetter (2009) tripower quarticity; BNKS (2010) semivariance.

VAR, cointegration, connectedness - quant_fund.models.var_coint:

- Sims (1980) VAR; Lutkepohl (2005); Pesaran, Shin (1998) generalized
  IRF/FEVD; Diebold, Yilmaz (2012) spillover index; Engle, Granger (1987);
  Johansen (1991/1995) trace/max-eig; MacKinnon-Haug-Michelis (1999)
  critical values; Granger (1969) causality F-tests; VECM reduced-rank.

Microstructure liquidity - quant_fund.features.liquidity:

- Amivest ratio; Lesmond, Ogden, Trzcinka (1999) LOT; Lesmond (2005) FHT;
  Pastor, Stambaugh (2003) gamma; Hasbrouck (1991) lambda; Glosten,
  Harris (1988) spread components; Holden (2009) effective tick.

Distribution and normality battery - quant_fund.metrics.distribution:

- Shapiro, Wilk (1965); Anderson, Darling (1954); Cramer-von Mises;
  Lilliefors (1967, MC p-values); Dagostino-Pearson (1973) K2; Pearson
  (1900) chi2; Mardia (1970) multivariate; Brys-Hubert-Struyf (2004)
  medcouple; Rousseeuw, Croux (1993) Qn scale.

Signal decomposition - quant_fund.models.decomposition:

- Broomhead, King (1986); Vautard-Ghil (1992); Golyandina-Nekrutkin-
  Zhigljavsky (2001) SSA + R-forecasting; Huang et al. (1998/2003) EMD,
  Hilbert-Huang instantaneous frequency; robust median-filter trend.

Calibration extensions - quant_fund.metrics.calibration2:

- Murphy (1973) Brier decomposition; Winkler (1972) interval score;
  Dawid, Sebastiani (1999); Hosmer, Lemeshow (1980); Murphy, Winkler
  (1977) reliability diagrams; Scheuerer, Hamill (2015) variogram score;
  pinball/check loss; Candille, Talagrand (2005) spread-skill.

Mixture models - quant_fund.models.mixture:

- Dempster, Laird, Rubin (1977) EM; McLachlan, Peel (2000); Peel,
  McLachlan (2000) robust t-mixtures (ECM with latent-scale weights);
  Schwarz (1978) BIC; Biernacki-Celeux-Govaert (2000) ICL-style entropy.

Pairs selection - quant_fund.models.pairs:

- Gatev, Goetzmann, Rouwenhorst (2006) distance approach; Vidyamurthy
  (2004) cointegration approach; Leung, Li (2016) OU optimal entry/exit;
  Chan (2013) half-life filters; Elliott-van der Hoek-Malcolm (2005).


## Wave 3 — Dynamic & nonlinear-inference canon

### Regime switching
- Hamilton, J.D. (1989). "A new approach to the economic analysis of nonstationary time series and the business cycle." *Econometrica* 57(2) — Markov-switching regression via EM + Hamilton filter.
- Kim, C.J. (1994). "Dynamic linear models with Markov-switching." *J. Econometrics* 60 — Kim smoother.
- Klaassen (2002). Improving GARCH volatility forecasts with regime-switching GARCH.

### Extended GARCH
- Ding, Granger & Engle (1993). "A long memory property of stock market returns and a new model." *J. Empirical Finance* 1 — APARCH.
- Baillie, Bollerslev & Mikkelsen (1996). "Fractionally integrated generalized autoregressive conditional heteroskedasticity." *J. Econometrics* 74 — FIGARCH.
- Engle & Ng (1993). "Measuring and testing the impact of news on volatility." *J. Finance* 48 — news impact curve.

### Forecast evaluation
- Clark & West (2007). "Approximately normal tests for equal predictive accuracy in nested models." *J. Econometrics* 138.
- Harvey, Leybourne & Newbold (1997). "Testing the equality of prediction mean squared errors." *IJF* 13 — HLN small-sample DM.
- Harvey, Leybourne & Newbold (1998). "Tests for forecast encompassing." *JBES* 16 — ENC-T.
- Giacomini & White (2006). "Tests of conditional predictive ability." *Econometrica* 74.
- Giacomini & Rossi (2010). "Forecast comparisons in unstable environments." *J. Applied Econometrics* 25 — fluctuation test.
- McCracken (2007). "Asymptotics for out of sample tests of Granger causality." *J. Econometrics* 140.

### Bootstrap & subsampling
- Wu (1986); Mammen (1993): wild bootstrap for heteroskedastic errors.
- Buhlmann (1997). "Sieve bootstrap for time series." *Bernoulli* 3.
- Politis & Romano (1994). "Large sample confidence regions based on subsamples." *Ann. Statist.* 22.
- MacKinnon (2009). Bootstrap hypothesis testing. In *Handbook of Computational Econometrics*.

### Panel econometrics
- Levin, Lin & Chu (2002). "Unit root tests in panel data." *J. Econometrics* 108 — LLC.
- Im, Pesaran & Shin (2003). "Testing for unit roots in heterogeneous panels." *J. Econometrics* 115 — IPS.
- Maddala & Wu (1999). "A comparative study of unit root tests with panel data." *Oxford Bull.* 61 — Fisher-ADF.
- Hadri (2000). "Testing for stationarity in heterogeneous panel data." *Econometrics J.* 3.
- Pesaran (2004/2015). "General diagnostic tests for cross section dependence in panels." *Cambridge WP* / *J. Applied Econometrics* — CD test.
- Anderson & Hsiao (1981). "Estimation of dynamic models with error components." *JASA* 76.
- Arellano & Bond (1991). "Some tests of specification for panel data." *Rev. Econ. Studies* 58 — diff-GMM + Sargan/AR(2).

### Nonlinear dependence
- Szekely, Rizzo & Bakirov (2007). "Measuring and testing dependence by correlation of distances." *Ann. Statist.* 35 — dCor.
- Gretton et al. (2005). "Measuring statistical dependence with Hilbert–Schmidt norms." *ALT* — HSIC.
- Gretton et al. (2012). "A kernel two-sample test." *JMLR* 13 — MMD.
- Chatterjee (2021). "A new coefficient of correlation." *JASA* 116 — xi_n.
- Kraskov, Stogbauer & Grassberger (2004). "Estimating mutual information." *Phys. Rev. E* 69 — KSG kNN MI.

### Kernel methods
- Rasmussen & Williams (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Scholkopf, Smola & Muller (1998). "Nonlinear component analysis as a kernel eigenvalue problem." *Neural Computation* 10 — kernel PCA.
- Shawe-Taylor & Cristianini (2004). *Kernel Methods for Pattern Analysis* — kernel ridge.
- Williams & Seeger (2001). "Using the Nystrom method to speed up kernel machines." *NeurIPS*.

### Event studies
- MacKinlay (1997). "Event studies in economics and finance." *J. Econ. Literature* 35.
- Brown & Warner (1985). "Using daily stock returns: the case of event studies." *J. Financial Econ.* 14.
- Patell (1976). "Corporate forecasts of earnings per share and stock price behavior." *J. Accounting Research* 14.
- Corrado (1989). "A nonparametric test for abnormal security-price performance." *J. Financial Econ.* 23.
- Boehmer, Musumeci & Poulsen (1991). "Event-study methodology under conditions of event-induced variance." *J. Financial Econ.* 30 — BMP.

### VaR backtesting
- Kupiec (1995). "Techniques for verifying the accuracy of risk measurement models." *J. Derivatives* 3.
- Christoffersen (1998). "Evaluating interval forecasts." *IER* 39.
- Haas (2001). "New methods in backtesting." In *Research in Finance*.
- Basel Committee on Banking Supervision (1996). *Supervisory framework for backtesting* — traffic-light zones.
- Berkowitz, Christoffersen & Pelletier (2011). "Evaluating value-at-risk models with desk-level data." *Management Science* 57.

### Long memory
- Geweke & Porter-Hudak (1983). "The estimation and application of long memory time series models." *JTSA* 4 — GPH.
- Kunsch (1987); Robinson (1995). "Gaussian semiparametric estimation of long range dependence." *Ann. Statist.* 23 — local Whittle.
- Fox & Taqqu (1986). "Large-sample properties of parameter estimates for strongly dependent stationary Gaussian time series." *Ann. Statist.* 14 — Whittle.
- Lo (1991). "Long-term memory in stock market prices." *Econometrica* 59 — modified R/S.

### Survival analysis
- Kaplan & Meier (1958). "Nonparametric estimation from incomplete observations." *JASA* 53.
- Nelson (1972); Aalen (1978): cumulative hazard estimation.
- Mantel (1966). "Evaluation of survival data and two new rank order statistics." *Cancer Chemo. Reports* 50 — log-rank.
- Cox (1972). "Regression models and life-tables." *JRSS-B* 34 — proportional hazards.
- Greenwood (1926): variance of the survival estimator.
