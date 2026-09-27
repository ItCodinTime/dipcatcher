# AUDIT P6.4 — model layer: distribution/vol family

Scope: `models/distribution.py, mixture.py, skew_t.py, skew_normal.py, qar.py,
regime.py, regime_dist.py, regime_switch.py, fhs.py, hstep.py, lgbm_q2.py,
conformal.py, conformal_dist.py, conformal_rank.py, localized_conformal.py,
weighted_conformal.py, posthoc_calibration.py, ensemble.py, garch_ext.py,
garch_midas.py, egarch.py, har.py, realized.py, realized_garch.py,
volatility.py, stoch_vol.py, rough_vol.py` — every file read end-to-end
against the spec named in its docstring.

ULTRAPLAN item: `P6.4 Model layer: every file in models/ vs its cited paper`
(docs/ULTRAPLAN_FRONTIER.md).

Verdicts: `clean` (checked, matches spec), `fixed` (real bug, patch + KAT),
`waived` (semantics intentionally looser than spec — documented).

| file | claim checked | verdict | fix commit |
|---|---|---|---|
| distribution.py | Hansen skew-t pdf/cdf/ppf splice, `select_wrappee_family_name` precedence, PIT/CRPS plumbing | clean: cdf splice at xi-inversion point correct; `np.clip` quantile bracket ±200σ is fail-closed (brentq raises on extreme tau); family-precedence parens verified intentional | — |
| mixture.py | EM/ECM for skew-t mixture incl. Peel–McLachlan ν-step; weight validation in `_mixture_params_1d` | clean: E-step responsibilities, M-step moM updates, ν clipped grid search all match | — |
| skew_t.py | Hansen (1994) skew-t pdf/cdf/ppf, constants `a,b,c` | clean: unit-variance normalization and CDF splice verified against paper algebra | — |
| skew_normal.py | Azzalini method-of-moments inversion (δ↦α) | clean: skewness-bound clamp and monotonic inversion correct | — |
| qar.py | Quantile autoregression (Koenker–Xiao) — QAR(p) quantile process | clean: pinball objective, lag construction causal | — |
| regime.py | `VolThresholdRegime`/`GaussianHMMRegime`/`SingleStateRegime` contracts | fixed: `VolThresholdRegime` predict-before-fit now raises RuntimeError (was cut=0.0); all-NaN vol fit now raises (was silent cut=NaN → all-low regime). `GaussianHMMRegime.aic_bic` unfitted raises RuntimeError (was AttributeError). Waived: `np.where(isfinite, x, 0)` predict-time feature imputation is the repo-wide convention (also in `LinearQuantile`, `TreeQuantile`, `LGBMQ2Distribution.predict`) — documented drift, not a bug. | audit commit |
| regime_dist.py | regime-conditional distribution wrapper; fallback exception scope | clean: fallback catches fit-time failures; `LinAlgError` propagates (fail-closed is correct — a matrix failure is not "regime absent"). | — |
| regime_switch.py | Hamilton filter + Kim smoother; EM P-update | clean: forward filter and smoothed ratios verified; P-update uses filtered-implied joints = documented EM approximation, not exact M-step (waived note) | — |
| fhs.py | Filtered historical simulation: GJR moment-matched ω, z=e/σ standardized residuals, skew-t MLE on z | clean: ω = v(1−β)−αv−γE[e²1{e<0}] verified; one-step GJR forecast causal; rearrange applied | — |
| hstep.py | iid Student-t h-step sum scaling σ√h + overlapping empirical sums | clean: unit-variance t normalization, `c[h:]−c[:−h]` overlap sums, per-construction monotone rearrange, block layout matches docstring | — |
| lgbm_q2.py | per-τ LGBM quantile boosters | clean: n_jobs=1 + fixed seed deterministic; non-finite predict-time features → 0.0 (house convention, waived as in regime.py) | — |
| conformal.py | split conformal + CQR; `expand_interval` | clean: `⌈(n+1)(1−α)⌉` order statistic; CQR score max{lo−y, y−hi}; inverted-interval swap is documented defensive normalization (waived note) | — |
| conformal_dist.py | conformal wrapper over distribution heads | clean: causal base/cal split, residual quantile wiring correct | — |
| conformal_rank.py | top-k conformal FDR guarantee | clean modulo note: `_date_fdp_cover` reports coverage 1.0 on dates where the oracle selects nothing — vacuous but consistent with set-coverage convention (waived) | — |
| localized_conformal.py | Lei–Wasserman/Guan RBF-weighted CQR; Kish ESS gate | fixed: ESS gate now computed on unclipped `rbf_weights` — `clip_weights` floored zeros to 1e-3, defeating the `min_ess` fallback for far-off-manifold queries. Waived: `clip_weights` itself keeps its asserted all-positive contract (test encodes it); clipped phantom mass in the weighted quantile is ≤ n_cal·1e-3 relative weight — negligible, flagged not fixed. | audit commit |
| weighted_conformal.py | Tibshirani LR-weighted split CQR; histogram density ratio | fixed: `bench_weighted_cqr` now emits `claim="research_metric_only"`, `dgp`, `seed` labels matching `bench_localized_cqr`/`bench_conformal_topk` (synthetic-output labeling gap). Quantile formula verified: normalized cumulative weight with 1/(Σw+1) mass at +∞; uniform weights reproduce split conformal. | audit commit |
| posthoc_calibration.py | isotonic/PIT recalibration | clean: monotone mapping, PIT histogram correction causal | — |
| ensemble.py | stacked distribution, covariance weights | clean modulo notes: `StackedDistribution` base/cal split causal; `covariance_weights` pinv makes the documented singular-fallback nearly unreachable and `rolling_inverse_mse` floors mse at 1e-18 instead of raising (waived — defensive ensemble semantics) | — |
| garch_ext.py | APARCH (DGE 1993) + FIGARCH (BBM 1996) | fixed(2): (a) `figarch_variance` λ recursion had the π-sign flipped — code used `λ_k = βλ_{k−1} + π_k − φπ_{k−1}` where the expansion of 1−(1−φL)(1−L)^d/(1−βL) requires `λ_k = βλ_{k−1} − π_k + φπ_{k−1}`; produced negative ARCH(∞) weights (verified against direct series expansion; at φ=β=0 the code yields λ_k=π_k<0 vs correct −π_k>0), with `max(acc,1e-12)` masking the resulting variance collapse. (b) `aparch_variance` silently carried forward non-finite/non-positive σδ from the previous step, letting divergent parameter draws produce finite paths inside QMLE — now raises FloatingPointError, caught by `nll` → 1e12 (same convention as `_egarch_path`→None). | audit commit |
| garch_midas.py | GARCH-MIDAS long/short-run decomposition | clean modulo note: trailing partial block is silently dropped from the MIDAS weighting (documented behavior; waived) | — |
| egarch.py | Nelson (1991) EGARCH + GJR (1993) Gaussian QMLE | fixed: both fits now reject `res.fun` ≥ 1e11/non-finite — previously an all-divergent Nelder-Mead run returned params with `loglik=-1e12` and no failure signal (sibling `realized_garch` already checks `result.success`). | audit commit |
| har.py | Corsi (2009) HAR-RV: daily/weekly(5)/monthly(22) cascades, Newey–West SEs | clean: regressors end at t−1 (causal), NW HAC Bartlett weights correct; log-domain `HARVol.predict` uses exp without smearing correction — known Jensen bias, modeling choice (waived) | — |
| realized.py | RV/BV/TPQ/BNS/Lee–Mykland/TSRV/realized-kernel estimators | fixed: comment fix only — Huang–Tauchen variance constant was commented "~2.467" (that is (π/2)² alone); the coded expression (π/2)²+π−5 ≈ 0.609 is correct. All estimator constants verified (μ1⁻², μ_{4/3}⁻³, Gumbel C_n/s_n, Parzen kernel, pre-averaging ψ constants). Waived note: TSRV last sub-return can span >K base returns when n mod K ≠ 0 (standard implementations differ; waived). | audit commit |
| realized_garch.py | Hansen–Huang–Shek (2012) log-linear Realized GARCH on daily Parkinson | clean: joint Gaussian NLL return+measurement terms verified; h_{t+1}=ω+β log h_t+γ log x_t ordering correct; unconditional anchor (ω+γξ)/(1−β−γφ) correct; Parkinson (logH/L)²/4ln2; explicit `returns=`/`realized_measure=` kwargs prevent label leakage; fallback path fail-closed with reason | — |
| volatility.py | EWMA/GARCH(1,1)/HAR/tree vol family + `GARCHVol` wrapper | clean: ARCH(∞)-boundary unit conversion and documented fallback verified; `_garch_snapshot` caches the result dict by reference (mutation risk — waived note, single-writer use) | — |
| stoch_vol.py | Harvey–Shephard (1996) lognormal SV via Kalman QMLE | fixed: `stoch_vol_fit` now rejects non-finite/≥1e11 `res.fun` (same missing-convergence-check class as egarch). Kalman update (F=P+R, K=P/F, Joseph-free scalar form), stationary init s_η²/(1−φ²), tanh |φ|<1 reparam, E[ln χ²]=−1.2704 and Var=π²/2 constants verified. | audit commit |
| rough_vol.py | GJR(2018) moment scaling, fOU variance curve, fOU simulator | fixed: `variance_curve_fit` rebound `lv` to log-variances then returned `lv[good]` as "lags" — callers received log-var values (e.g. [0.04, 0.76, 1.19] instead of [1,2,3]) and any non-positive var_d triggered an IndexError (boolean mask longer than the rebound array). Now returns true lag values. `logvol_hurst`/`rough_signature` scaling regressions verified. | audit commit |

## Summary

- Real bugs fixed: 8 (figarch λ sign; aparch divergent-path carry-forward;
  variance_curve_fit lags mislabeling + IndexError; localized-conformal ESS
  gate on clipped weights; bench_weighted_cqr missing labels; egarch/gjr +
  stoch_vol missing convergence checks; VolThresholdRegime unfitted/NaN-cut
  silently predicting; GaussianHMMRegime.aic_bic wrong exception type).
- Regression tests: tests/unit/models/test_p64_dist_vol_audit.py — 13 KATs
  (impulse-extraction λ check vs direct series expansion; pure-fracdiff
  limit; divergence raises; lag-value identity; ESS-gate honesty;
  label presence; monkeypatched garbage-objective guards; fitted-state
  guards).
- Waivers (documented looser semantics, not fixed): predict-time
  NaN→0 feature imputation (repo-wide), `clip_weights` positive floor,
  vacuous coverage on empty-oracle dates, filtered-joint EM approximation,
  singular-cov pinv fallback, mse floor, MIDAS trailing block, log-domain
  HAR smearing, TSRV ragged last sub-return, skew-t ppf ±200σ bracket.
- No evidence of look-ahead leakage, wrong estimator constants, or
  score-contract (proper-score) violations in the audited files.
