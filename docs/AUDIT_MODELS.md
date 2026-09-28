# Models audit — `src/quant_fund/models/`

Deep correctness audit of the estimator/model catalog (141 modules, ~28k
LOC). Contract under test: fail-closed causality and honesty — look-ahead,
silent NaN propagation, fabrications, or non-contract error handling are
defects.

Method: (a) estimator-family formula checks against docstring/cited papers;
(b) causality — index math touching future rows, rolling/expanding windows
including the prediction row, statistics fit on full sample; (c) fail-closed —
degenerate inputs (constant series, NaN/inf, empty, wrong shape) must raise or
return labeled NaN, never silently produce numbers; (d) numerical
overflow/underflow paths (exp/log space, covariance inversions, optimizer
invalid-region penalties). Every fixed defect has a regression test in
`tests/unit/models/test_models_audit.py`.

Legend: **FIXED** — defect found, patched, regression-tested. **CLEAN** —
audited paths verified (guards, index math, formula spot-check); no defect.
**DEFERRED** — coverage shallow; deeper verification pending or fix deferred.

## Fixed defects

| Module | Verdict | Evidence |
|---|---|---|
| ar_estimation | FIXED | `levinson_durbin` floored `sigma2` at 1e-12 when the autocovariance sequence was not positive-definite (`|k|>=1`); now raises ValueError. |
| arma | FIXED | `arma_css` accepted a BFGS result at the 1e14 non-finite penalty as an optimum; now raises. |
| archimedean_extra | FIXED | `joe_fit` accepted `minimize_scalar` landing on the 1e12 penalty (returned theta=bound, loglik=-1e12); now raises. |
| caviar | FIXED | multi-start loop could keep `res.fun == 1e12` (invalid-parameter penalty) as best, then `_q_path` returned None into an `assert`; penalty now excluded, raises on all-fail. |
| cs_papers | DEFERRED | Real boundary-overlap: `MSFECombinationRanker.fit` nested split `groups[:n_train]`/`groups[n_train:]` has no embargo; with forward labels (`future_*_N`, N>1) the last ~N train date-groups' label windows straddle the hold groups, contaminating inner MSFE weights. Fix deferred — central purge sweep in progress. |
| dcc | FIXED | stage-2 Nelder-Mead accepted the 1e12 penalty when `a+b < 0.999`; now raises. |
| diffusion_index | FIXED | h-step forecast evaluated the design row at `t = T-1-h` (last fitted row) instead of the true origin `t = T-1`; forecast was stale by h steps. |
| duration | FIXED | `_nll_weibull` missed the `(gamma-1)*ln(1/psi)` term in the eps->x density transform; psi coefficient was -1 instead of -gamma, biasing WACD fits for gamma != 1. |
| dynamic_panel | FIXED | `ar2_coef` computed the lag-1 correlation of differenced residuals; the Arellano-Bond AR(2) test needs lag-2. |
| factor_models | FIXED | `bai_ng_factors` on a zero-variance panel divided by zero and silently returned `r_max-1`; now raises. |
| fgls_ar1 | FIXED | `_rho_from_resid` could return `|rho| >= 1` (denominator over lagged-only terms), making the Prais-Winsten transform `sqrt(1-rho^2)` invalid; now raises. |
| filters | FIXED | `ravn_uhlig_lambda` docstring stated `1600*(4/ppy)^4` (inverted); code correctly computes `1600*(ppy/4)^4`. Docstring corrected. |
| garch_ext | FIXED | `fit_figarch`/`fit_aparch` accepted a 1e12 penalty as `best.fun`; now raises. |
| garch_midas | FIXED | `garch_midas_fit` checked only finiteness of `res.fun`; a 1e12 penalty passed and garbage theta_hat was returned. Now raises. |
| gas | FIXED | `gas_vol_forecast` always applied the `inv_sqrt` update `s/sqrt(info)`; `scaling="unit"` fits were forecast with the wrong recursion. `gas_vol_fit` now returns `scaling` and the forecast honors it. |
| gmm_est | FIXED | both GMM steps accepted `res.fun == 1e14` penalty objectives; now raise. |
| kernel | FIXED | `fit_gp_regression` hyperparameter optimization accepted a 1e12 penalty (`np.isfinite` only); now falls back to defaults like a failed fit. |
| long_memory | FIXED | `local_whittle`/`whittle_arfima` returned `d` from an arbitrary bound point when every evaluation hit the 1e12 penalty; now raise. |
| nonlinear_filters | FIXED | `particle_filter` reset `w = exp(lw)` each step, discarding previous importance weights between resampling events; also corrected the marginal-likelihood increment to `log(sum w_prev * exp(lw))`. |
| nowcasting | FIXED | `fit_midas` with `ar_lag=False`: `yl[0]` is NaN, so `c*yl` poisoned every SSE evaluation to 1e12 — the optimizer never actually optimized and silently returned seed params. Also added the penalty guard on `best.fun`. |
| realized_garch | FIXED | a Nelder-Mead/L-BFGS-B result at the 1e12 penalty passed the success/finiteness check; now rejected (returns None, fail-closed). |
| skew_t | FIXED | `skew_t_fit` had no post-fit check; a 1e12 penalty produced garbage `nu/lam/mu/sigma` and `loglik=-1e12`; now raises. |
| state_space | FIXED | `ou_mle` silently clipped `phi` to `[1e-6, 0.999999]`, fabricating a finite mean-reversion speed on non-stationary series; phi outside (0,1) now raises. |
| tail | FIXED | `ScaledHistoricalTail.predict_var_es` on an unfitted model returned zeros (`0*scale`); `DrawdownClassifier.predict_proba` unfitted returned 0.05 for every row. Both now raise RuntimeError. |
| term_structure | FIXED | `svensson_fit` accepted the 1e12 penalty from `ssr_of` (invalid lambda region); now raises. |
| var_coint | FIXED | four defects: (1) `engle_granger` checked y but not x finiteness — NaN x silently propagated through lstsq; (2) `johansen_test` CVs were indexed by `r` instead of `n-r`, over-rejecting r=0 (used the 1-dim CV 3.76 instead of the n-dim entry); (3) `rank_trace`/`rank_max` counted non-rejections instead of the first non-rejected r (Johansen sequential rank); (4) `vecm_fit` reshaped `bf[rank:]` as `(n, p-1, n).transpose(1,0,2)`, scrambling lag blocks — corrected to `(p-1, n, n).transpose(0,2,1)`. |

## Full module table

| Module | Verdict | Evidence |
|---|---|---|
| agaci | CLEAN | Asymmetric-GARCH quantile recursion and guards reviewed; degenerate inputs raise. |
| alpha | CLEAN | Alpha/signal combination logic reviewed; weight normalization guards present. |
| american_baw | DEFERRED | Barone-Adesi-Whaley quadratic approximation — surface review only; boundary/early-exercise edge cases not deeply verified. |
| ar_estimation | FIXED | See above. YW/Burg/OLS AR paths otherwise consistent; AR roots checked. |
| arch_fit | CLEAN | ARCH/GARCH QMLE loop reviewed; variance path positivity and convergence guards present. |
| archimedean_extra | FIXED | See above (joe_fit penalty). Frank/Gumbel cdf-pdf spot-checked. |
| arma | FIXED | See above (CSS penalty). Residual recursion indexing verified causal. |
| asset_pricing | CLEAN | `date_groups` first-seen/sorted grouping reviewed; Fama-MacBeth/panel stats guards verified. |
| bachelier | DEFERRED | Bachelier pricing formulas spot-checked only; discretization edges not deeply verified. |
| bandits | DEFERRED | Arm selection loops reviewed at surface; per-arm posterior update causality not deeply verified. |
| base | CLEAN | Protocols/mixins reviewed; no estimation logic. |
| bayesian | CLEAN | Conjugate update formulas verified against docstrings; prior/posterior guards present. |
| bond_analytics | CLEAN | Discount/YTM/duration formulas spot-checked; day-count guards present. |
| calibration | CLEAN | Isotonic/logistic calibration paths reviewed; degenerate-class handling raises/falls back honestly. |
| carr_madan | DEFERRED | FFT pricer surface-reviewed; truncation/damping edge behavior not deeply verified. |
| caviar | FIXED | See above (penalty acceptance). CaViaR recursion causal. |
| ccm | CLEAN | Convergent cross-mapping shadow-manifold logic reviewed; library/embedding guards present. |
| changepoint | CLEAN | PELT/BOCPD paths reviewed; segment boundary indexing verified causal. |
| conformal | CLEAN | Split-conformal quantile math verified; calibration/prediction split respected. |
| conformal_dist | CLEAN | Weighted/local conformal variants reviewed; score quantile indexing verified. |
| conformal_rank | CLEAN | Rank-based conformal reviewed; exclusive/inclusive quantile conventions consistent. |
| copula | CLEAN | Gaussian/t copula fit + simulation reviewed; pseudo-observation rank transform causal. |
| count | CLEAN | Poisson/NB count GLMs reviewed; link positivity guarded. |
| covariance | CLEAN | Ledoit-Wolf/OAS/shrinkage family deeply reviewed — trailing-complete-window causality, PSD repair, labeled fallback. |
| crc | CLEAN | Conformal risk control reviewed; monotone-risk and monotone-loss guards present. |
| creditrisk_plus | DEFERRED | CreditRisk+ banding recursion spot-checked; edge behavior on degenerate portfolios not verified. |
| croston | CLEAN | Croston/SBA intermittent-demand recursion reviewed; zero-demand handling guarded. |
| cs_papers | DEFERRED | See above — nested MSFE train/hold boundary overlap documented; remainder of module swept (guards, finite-checks). |
| cv_plus | CLEAN | CV+/jackknife+ residual-weight formulas reviewed; normalization guards present. |
| dcc | FIXED | See above (stage-2 penalty). Q recursion and correlation extraction verified. |
| decomposition | CLEAN | STL/MSTL-style decomposition reviewed; window boundary handling causal. |
| deep_rl | DEFERRED | Single `nan_to_num` use noted (reward shaping); env-loop causality reviewed at surface only. |
| dfm | CLEAN | Dynamic factor model — EM/Kalman paths reviewed; unstable-A Lyapunov guard verified; PC scores/RTS smoother correct. |
| diffusion_index | FIXED | See above (forecast origin off-by-h). Stock-Watson factor extraction verified. |
| discrete | CLEAN | Discrete choice/logit paths reviewed; utility guards present. |
| distribution | CLEAN | Distribution heads (gauss/t/empirical quantile paths) reviewed; unfitted raises. |
| dlinear | DEFERRED | Neural head — torch internals surface-reviewed; normalization-window causality needs torch-level verification. |
| duration | FIXED | See above (WACD Weibull Jacobian). Exp-ACD loglik verified; psi path causal. |
| dynamic_panel | FIXED | See above (AR(2) lag). GMM instrument matrix construction verified causal. |
| edgeworth | CLEAN | Edgeworth expansion cumulant recursion verified; truncation guards present. |
| efficient_frontier | CLEAN | Markowitz frontier/GMV formulas verified; PSD guards on inversion. |
| egarch | CLEAN | EGARCH QMLE reviewed — `res.fun >= 1e11` penalty guard already present; log-variance path guarded. |
| elliptical | CLEAN | Elliptical (t/EPD) density and tail formulas spot-checked. |
| empirical_copula | CLEAN | Empirical copula ranks reviewed; pseudo-observation transform causal. |
| enbpi | CLEAN | Ensemble bootstrap prediction intervals reviewed; OOB residual causality intact. |
| ensemble | CLEAN | Vincentization/quantile-panel combination reviewed; weight normalization guarded. |
| ets | CLEAN | ETS (error-trend-seasonal) recursion reviewed; degenerate inputs raise. |
| event_study | CLEAN | Event-window alignment reviewed; abnormal-return estimation windows verified causal. |
| expectile | CLEAN | ALS expectile iteration verified; tau guards present. |
| factor_models | FIXED | See above (bai_ng zero-variance). PCA/Fama-MacBeth/gics paths reviewed. |
| fgls_ar1 | FIXED | See above (rho bounds). PW/CO transform indexing verified. |
| fhs | CLEAN | Filtered historical simulation reviewed; scale/sigma causality verified. |
| filters | FIXED | See above (docstring). HP/BK/Hamilton filter boundary handling verified. |
| fracdiff | CLEAN | Fractional differencing weights recursion verified; window causality intact. |
| fractional | CLEAN | Fractional calculus helpers reviewed; fixed-window truncation guarded. |
| functional | CLEAN | Functional-regression basis paths reviewed; quadrature guards present. |
| gam | CLEAN | GAM spline/p-spline basis reviewed; penalty matrix guards present. |
| garch_ext | FIXED | See above (FIGARCH/APARCH penalties). Variance-path divergence raises. |
| garch_midas | FIXED | See above (penalty). MIDAS weight normalization verified. |
| gas | FIXED | See above (unit-scaling forecast). Score/info formulas verified for gauss/t. |
| gaussian_copula_default | CLEAN | Gauss-Hermite quadrature for M~N(0,1) verified correct against exact bivariate-normal moments (probabilists' He weights w/sqrt(2pi), raw nodes). |
| glasso | CLEAN | Graphical lasso paths reviewed; PSD/regularization guards present. |
| gmm_est | FIXED | See above (penalty checks on both steps). NW HAC covariance verified. |
| har | CLEAN | HAR-RV restricted lag structure verified; causality intact. |
| hedging | CLEAN | Delta/greek hedge helpers reviewed; degenerate-spot guards present. |
| hstep | CLEAN | Horizon-step forecast wrapper reviewed; lag alignment verified causal. |
| information_share | CLEAN | Hasbrouck/GK information-share formulas verified; Cholesky ordering documented. |
| iv_approx | DEFERRED | Implied-vol approximations surface-reviewed; inversion edge cases not deeply verified. |
| jackknife_plus | CLEAN | Jackknife+ leave-one-out quantile math verified. |
| johnson_sb | CLEAN | Johnson SB transform/quantile formulas spot-checked; parameter guards present. |
| johnson_su | CLEAN | Johnson SU fit paths reviewed; bounded-domain handling guarded. |
| jump_diffusion | DEFERRED | Jump-diffusion moments/compound-Poisson spot-checked; jump-censor edges not verified. |
| kernel | FIXED | See above (GP hyperparam penalty). Kernel-ridge/KPCA guards verified. |
| kmv | DEFERRED | KMV distance-to-default solver surface-reviewed; equity/debt edge cases not verified. |
| kronos | DEFERRED | Neural head — surface review only; embedding/temporal masking causality not verified at torch level. |
| leisen_reimer | DEFERRED | Binomial-tree option pricing surface-reviewed; odd/even node parity not verified. |
| lgbm_q2 | DEFERRED | LightGBM quantile head — wrapper surface-reviewed; leaf-empirical quantile path not verified. |
| local_projection | CLEAN | Jorda local-projection horizon indexing verified causal (lead-lag construction). |
| localized_conformal | CLEAN | Localized conformal scores reviewed; kernel-weight normalization guarded. |
| long_memory | FIXED | See above (Whittle penalty x2). GPH/Lo-modified-RS reviewed. |
| lowess | CLEAN | LOWESS tricube weighting and window bounds verified. |
| market_making | DEFERRED | Avellaneda-Stoikov quotes surface-reviewed; inventory skew edges not verified. |
| mfdfa | CLEAN | MFDFA fluctuation functions reviewed; profile causality verified. |
| mixture | CLEAN | Gaussian-mixture EM reviewed; responsibility degenerate guards present. |
| momentum | CLEAN | Momentum feature transforms reviewed; window bounds causal. |
| nbeats | DEFERRED | Neural head — basis/block surface-reviewed; forecast-window alignment needs torch-level verification. |
| ngboost_lite | DEFERRED | NGBoost-lite natural-gradient head — surface-reviewed; scoring-rule gradient paths not verified. |
| nig_vg | DEFERRED | NIG/VG characteristic-function and pdf paths spot-checked; tail quadrature edges not verified. |
| nonlinear_filters | FIXED | See above (particle weight carry + loglik increment). EKF/UKF sigma-point paths verified. |
| nowcasting | FIXED | See above (MIDAS ar_lag NaN poisoning + penalty). Bridge equation alignment verified. |
| online_crc | CLEAN | Online conformal risk control reviewed; adaptive lambda updates causal. |
| options | DEFERRED | Option pricing wrapper — surface-reviewed; exercise-style dispatch edges not verified. |
| ordered | CLEAN | Ordered probit/logit threshold estimation reviewed; cutpoint guards present. |
| pairs | CLEAN | Pairs-trading spread/half-life helpers reviewed; hedge-ratio estimation causal. |
| panel_coint | CLEAN | Panel cointegration (Pedroni-style) stats reviewed; cross-sectional pooling guards present. |
| poet | CLEAN | POET principal-orthogonal thresholding verified; eigenvalue ordering correct. |
| point_process | CLEAN | Hawkes intensity recursion verified; positivity guards present. |
| posthoc_calibration | CLEAN | Post-hoc calibration (isotonic/Platt) reviewed; unfitted raises. |
| qar | CLEAN | Quantile AR recursion verified; monotone-quantile guards present. |
| qrf | DEFERRED | Quantile random-forest leaf aggregation — surface-reviewed; leaf-weight normalization not verified. |
| quantile_bandit | DEFERRED | Quantile-bandit arm posteriors — surface-reviewed; empirical-CDF update causality not verified. |
| quantile_forest | CLEAN | Quantile-forest weighting scheme reviewed; leaf-store causality verified. |
| random_projection | CLEAN | Johnson-Lindenstrauss projection reviewed; component guards present. |
| ranking | CLEAN | Ranking head dispatch reviewed; per-date group scoring verified. |
| realized | CLEAN | Realized variance/covariance estimators (RV, bipower, kernel) verified. |
| realized_garch | FIXED | See above (penalty reject -> None). Measurement-equation filtering reviewed. |
| regime | CLEAN | HMM/regime detection paths reviewed; state-sequence causality verified. |
| regime_dist | CLEAN | Regime-conditional distribution heads reviewed; per-regime guards present. |
| regime_switch | CLEAN | Markov-switching recursions verified; filtered/smoothed indexing causal. |
| rl | DEFERRED | RL environment loop — surface-reviewed; reward/observation timing not deeply verified. |
| rmt | CLEAN | Random-matrix-theory MP bounds and eigenvalue clipping verified. |
| robust_cov | CLEAN | MCD/M-estimator robust covariance reviewed; reweighting guards present. |
| rough_vol | DEFERRED | Rough-volatility (rBergomi-style) — surface-reviewed; Volterra kernel edges not verified. |
| sabr | DEFERRED | SABR implied-vol expansion reviewed at surface (bisection bound check present); smile wings not verified. |
| seasonal | CLEAN | Seasonal decomposition/dummies reviewed; period guards present. |
| short_rate | DEFERRED | Short-rate model paths (Vasicek/CIR-style) surface-reviewed; discretization edges not verified. |
| skew_normal | CLEAN | Skew-normal pdf/cdf/ppf spot-checked; shape guards present. |
| skew_t | FIXED | See above (fit penalty). Density/quantile paths verified. |
| smoothers | CLEAN | Kernel/spline smoothers reviewed; bandwidth guards present. |
| sparse | CLEAN | Sparse-regression (lasso/elastic-net) paths reviewed; screening guards present. |
| spectral | CLEAN | Spectral density/periodogram estimators reviewed; taper/window causality verified. |
| spectral_clustering | CLEAN | Spectral clustering embedding reviewed; normalized-Laplacian guards present. |
| stable | CLEAN | Stable-distribution characteristic/numerical inversion paths reviewed. |
| state_space | FIXED | See above (ou_mle phi). Kalman predict/update indexing verified causal. |
| stoch_vol | CLEAN | SV QMLE reviewed — `res.fun >= 1e11` penalty guard already present; log-vol path guarded. |
| survival | CLEAN | Kaplan-Meier/Cox-style survival estimators reviewed; censoring guards present. |
| tail | FIXED | See above (unfitted fabrication x2). POT/GPD exceedance paths verified. |
| tempered_stable | CLEAN | Tempered-stable characteristic/density paths reviewed. |
| term_structure | FIXED | See above (Svensson penalty). NS loadings verified. |
| theta | CLEAN | Theta-method SES + drift decomposition verified. |
| threshold | CLEAN | Threshold AR/SETAR regime splitting reviewed; delay-index causality verified. |
| trade_sign | CLEAN | Lee-Ready/Easdley-O'Hara trade-signing rules verified. |
| tukey_gh | CLEAN | Tukey g-and-h quantile/inversion paths reviewed. |
| var_coint | FIXED | See above (4 defects). VAR/Granger/FEVD/Diebold-Yilmaz indexing verified. |
| variance_swap | CLEAN | Variance-swap replication formula reviewed; fair-strike integration guarded. |
| vasicek_credit | DEFERRED | Vasicek large-portfolio credit formulas surface-reviewed; correlation clamp edges not verified. |
| volatility | CLEAN | EWMA/Parkinson/GK/RS vol estimators verified; window causality trailing-complete. |
| watch | CLEAN | Watchdog/unfitted guards reviewed. |
| wavelets | CLEAN | Wavelet transform boundary handling and MODWT-style causality verified. |
| weighted_conformal | CLEAN | Weighted conformal reviewed; covariate-shift weight normalization guarded. |
| whitening | CLEAN | PCA/ZCA whitening reviewed; eigenvalue-floor guards present. |

Note: `__init__.py`, `__init__.pyi`-style module files and pure-reexport
shims are excluded (no estimation logic).
