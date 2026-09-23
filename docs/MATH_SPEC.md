**Day Wave 45 contract (closed-form Student-t CRPS):** Research-only `crps_student_t` / `mean_crps_student_t` (Jordan–Krüger–Lerch / scoringRules; ν>2 fail-closed NaN; bad σ → NaN). Distribution bench surfaces `crps_scaled_student_t_closed` beside quantile Riemann `crps_scaled_student_t` on scaled `vol_20` path. Dual view — not a live capital claim. (Numbered Wave45: sibling claimed day_grind DayWave43 for `assume_sorted` history prefix and parked Wave44 soft scorecard forge.)

**Day Wave 42 contract (Jackknife+/CV+ coverage_guarantee_scope receipt verify):** Soft research-receipt honesty parallel to VaR/ES battery key-presence gates: when nonempty `families['jackknife_plus']` or `families['cv_plus']` exposes `coverage` **or** `coverage_floor` (key present — values may be NaN), `verify_research_artifact` requires `coverage_guarantee_scope == "marginal_exchangeable"` (Wave41 honesty key). Missing key → `coverage_guarantee_scope_missing:<fam>`; wrong value → `coverage_guarantee_scope_invalid:<fam>`. Empty `{}` / no coverage keys → skip. Helpers: `research.catalog.jp_cv_blob_requires_marginal_coverage_scope` / `coverage_guarantee_scope_is_marginal` / `coverage_guarantee_scope_consistency_errors`. Research-diagnostic only — **never** a live capital / promotion gate; does **not** reopen soft-verify H-table sprawl (H5/H6/H13/H14 remain paused). No forged soft scorecard flag.

**Day Wave 44 contract (soft scorecard executed/nonempty/finite_observation forge honesty):** Soft research-receipt honesty parallel to `forbidden_metrics_absent` / battery `*_ok` forge gates: when a scorecard family claims `executed` / `nonempty` / `finite_observation` **True** but catalog helpers disagree (`family_blob_executed` / `family_blob_nonempty` / `family_blob_has_finite_observation` — lifted from agent nested `has_observation`), `verify_research_artifact` emits `scorecard_executed_flag_forged:<fam>` / `scorecard_nonempty_flag_forged:<fam>` / `scorecard_finite_observation_flag_forged:<fam>`. False/absent flags are **not** forgeries (`scorecard_invalid` still applies for valid receipts). Catches hand-edited receipts that claim those flags True on `{}` or all-NaN blobs. Research-diagnostic only — **never** a live capital / promotion gate; does **not** reopen soft-verify H-table sprawl (H5/H6/H13/H14 remain paused).

**Day Wave 43 contract (`assume_sorted` history prefix fail-closed):** `history_upto` / `history_for_calibration` / `optimize_asof` trailing-hist path take `history_prefix_upto` only under `HISTORY_SORT_KEYS` (or after `assume_sorted=True` **plus** one `under_history_sort_contract` check). A nonempty frame with `event_time` that violates the contract while `assume_sorted=True` raises `ValueError` — lying callers must not binary-search an unsorted series. Empty frames OK. No-`day_index` path remains plain `filter(event_time <= asof)`. Happy-path cost: one contract check when `assume_sorted=True`. Research/engineering honesty — not a live P&L claim.




**Day Wave 18 contract (VaR-battery verify):** Soft research-verify honesty: when nonempty `families['tail']` contains `kupiec_p` **or** `kupiec_lr` (values may be NaN), `verify_research_artifact` requires presence of `christoffersen_cc_p`/`christoffersen_cc_lr` and preferred `christoffersen_ind_p`/`christoffersen_ind_lr` (values may be NaN). Empty `{}` skips. Missing keys → `tail_var_battery_incomplete:<key>` (research-receipt fail-closed). Helpers: `research.catalog.tail_var_battery_missing_keys` / `tail_var_battery_keys_present`; soft scorecard `tail_var_battery_ok` on the `tail` family — **never** a live capital / promotion gate. (Separately, Day Wave 18 also wires distribution-bench `e_dm_crps_*` — see CRPS section.)



**Day Wave 21 contract (ES-battery verify):** Soft research-verify honesty parallel to VaR-battery: when nonempty `families['tail']` contains `es_95` **or** `realized_es` **or** `var_95` (values may be NaN), `verify_research_artifact` requires presence of `acerbi_szekely_z1`/`acerbi_szekely_z2`/`fissler_ziegel_mean`/`es_hit_count` (values may be NaN). Empty `{}` skips. Missing keys → `tail_es_battery_incomplete:<key>`. Helpers: `research.catalog.tail_es_battery_missing_keys` / `tail_es_battery_keys_present`; soft scorecard `tail_es_battery_ok`. **Orthogonal** to VaR-battery — does not double-require Christoffersen; when both Kupiec and ES markers present, both batteries apply independently. Never a live capital / promotion gate.

**Day Wave 25 contract (soft battery_ok forge fail-closed; day_grind DayWave23):** Parallel to `scorecard_forbidden_flag_forged`: when scorecard claims `tail_var_battery_ok` / `tail_es_battery_ok` / `dist_crps_eprocess_ok` is **True** but `tail_var_battery_keys_present` / `tail_es_battery_keys_present` / `dist_crps_eprocess_keys_present` on the family blob is False, verify appends `scorecard_tail_var_battery_flag_forged:tail` / `scorecard_tail_es_battery_flag_forged:tail` / `scorecard_dist_crps_eprocess_flag_forged:distribution`. Absent/False `*_ok` are not forgeries; existing `*_incomplete:<key>` errors remain. Honest agent path sets flags via helpers — forged True must not pass. Research diagnostic only — never a live capital / promotion gate. (Numbered Wave25 in SOTA to avoid colliding with Wave23 promotion-receipt / Wave24 run_id sections; progress banner keeps DayWave23.)

# Mathematical specification

Conventions used in code. If an academic method is modified, the modification is stated here.

## Returns

Simple return:

\[
R_t = \frac{P_t}{P_{t-1}} - 1
\]

Log return:

\[
r_t = \ln(P_t / P_{t-1})
\]

Forward simple return over horizon \(h\) trading periods:

\[
y_{i,t,h} = \frac{P_{i,t+h}}{P_{i,t}} - 1
\]

Alpha research uses **total-return** simple returns (dividends reinvested). Volatility and distribution engines use **log total returns**. Raw unadjusted prices are stored and never overwritten.

Benchmark excess:

\[
\alpha^{\text{target}}_{i,t,h} = R_i(t,t+h) - R_{\text{benchmark}}(t,t+h)
\]

Sector-relative:

\[
R_i - R_{\text{sector}}
\]

## Pinball loss

For quantile \(\tau \in (0,1)\) and forecast \(q\):

\[
L_\tau(y,q) = \begin{cases}
\tau (y-q) & y \ge q \\
(1-\tau)(q-y) & y < q
\end{cases}
= \max\bigl(\tau(y-q),\,(\tau-1)(y-q)\bigr)
\]

Mean pinball is the average of \(L_\tau\) over observations. Crossing rate is the fraction of rows with \(Q_{\tau_1} > Q_{\tau_2}\) for some \(\tau_1 < \tau_2\). Optional repair: sort quantiles (rearrangement). Crossing rate is still reported on the **raw** forecasts.

## CRPS approximation from quantiles

Given strictly increasing quantile levels \(\tau_k\) and forecasts \(q_k\), approximate the CRPS of a piecewise-linear CDF by the Riemann sum of pinball losses (Gneiting & Raftery, 2007):

\[
\widehat{\mathrm{CRPS}}(F,y) = 2\sum_{k=1}^{K} L_{\tau_k}(y,q_k)\,(\tau_k-\tau_{k-1})
\]

with \(\tau_0=0\) (the factor 2 is the Gneiting–Raftery quantile-decomposition
identity; without it the sum reports half-CRPS). This is an approximation,
not the closed-form CRPS of a parametric law.


## Closed-form Gaussian CRPS

For a Gaussian predictive \(F = N(\mu,\sigma^2)\) with \(\sigma>0\) (Gneiting & Raftery, 2007):

\[
\mathrm{CRPS}\bigl(N(\mu,\sigma^2), y\bigr)
= \sigma\Bigl[z\,(2\Phi(z)-1) + 2\varphi(z) - \frac{1}{\sqrt{\pi}}\Bigr],
\quad z=\frac{y-\mu}{\sigma}
\]

where \(\varphi\)/\(\Phi\) are the standard normal pdf/cdf. At \((\mu,\sigma,y)=(0,1,0)\) this
equals \((\sqrt{2}-1)/\sqrt{\pi}\) (not \(1/\sqrt{\pi}\)). Lab helpers:
`crps_gaussian` (elementwise) / `mean_crps_gaussian`. Non-positive or non-finite
\(\sigma\) → honest NaN. Research-diagnostic only — never a live capital claim.

Distribution bench (`bench_distribution` / `_distribution_horizon_scores`) reports research-only `crps_gaussian_closed` (and `crps_scaled_gaussian_closed` when vol scale is available) **beside** the existing quantile Riemann keys `crps_gaussian` / `crps_empirical` / `crps_scaled_gaussian` — dual view, not a replacement. Optional `dm_crps_*` is Diebold–Mariano on per-observation quantile-CRPS losses (gaussian vs empirical holdout). Day Wave 18 adds research-only `e_dm_crps_final` / `e_dm_crps_reject` / `e_dm_crps_n` via Wave4 `e_process_dm` on the same CRPS loss differential (prefixed to avoid vol-bench `e_dm_*` key clash; Day Wave 20: NaN/False/0 presence sentinels on e-process failure); not a live capital claim. Day Wave 19, when the scaled `vol_20` path fits both ScaledGaussian (`qs`) and ScaledStudentT (`qt`), also surfaces research-only `dm_crps_scaled_preferred` / `dm_crps_scaled_p` / `dm_crps_scaled_stat` plus `e_dm_crps_scaled_final` / `e_dm_crps_scaled_reject` / `e_dm_crps_scaled_n` via the same helpers on per-obs quantile-CRPS losses (gate `n_te>=3`; on e-process failure emit NaN/False/0 presence sentinels — Day Wave 20); diagnostics only — does not change wrappee selection; not a live capital claim. Day Wave 20 soft research-receipt verify: nonempty distribution with `dm_crps_p` / `dm_crps_scaled_p` must expose matching `e_dm_crps_*` / `e_dm_crps_scaled_*` key presence (values may be NaN); empty `{}` skips; not a live promotion gate.

## Empirical (ensemble) CRPS

Given draws \(X_1,\ldots,X_n\) from the predictive and a single observation \(y\):

\[
\widehat{\mathrm{CRPS}}
= \frac{1}{n}\sum_{i=1}^{n}|X_i-y|
- \frac{1}{2n^2}\sum_{i=1}^{n}\sum_{j=1}^{n}|X_i-X_j|
\]

Implemented as `crps_empirical` (research-only; empty / all-non-finite sample → NaN).

## Closed-form Student-t CRPS

For a location-scale Student-t predictive with location \(\mu\), scale \(\sigma>0\),
and degrees of freedom \(\nu>2\) (Jordan, Krüger & Lerch / scoringRules; Gneiting &
Raftery CRPS). Let \(z=(y-\mu)/\sigma\) and \(f_\nu\)/\(F_\nu\) the standard-t pdf/cdf:

\[
\mathrm{CRPS}(F_{\mu,\sigma,\nu}, y)
= \sigma\Biggl[
  z\,(2F_\nu(z)-1)
  + \frac{2}{\nu-1}\Bigl(
    f_\nu(z)\,(\nu+z^2)
    - \sqrt{\nu}\,
    \frac{B(\tfrac12,\nu-\tfrac12)}{B(\tfrac12,\tfrac{\nu}{2})^2}
  \Bigr)
\Biggr]
\]

Lab helpers: `crps_student_t` (elementwise) / `mean_crps_student_t`. Contract:
\(\nu\le 2\) or non-finite → honest NaN (stricter than the finite-mean \(\nu>1\)
domain — fail-closed with finite variance); non-positive/non-finite \(\sigma\) → NaN;
empty → []/NaN; length mismatch → ValueError. Research-diagnostic only — never a
live capital claim.

Distribution bench (`_distribution_horizon_scores`) reports research-only
`crps_scaled_student_t_closed` **beside** quantile Riemann `crps_scaled_student_t`
when the scaled `vol_20` Student-t path is present (Day Wave 45; sibling race took
Wave 43 for `assume_sorted` history prefix) — dual view, not a replacement.
No soft-verify H-table sprawl; not a live capital claim.


## QLIKE

Reported on **variance** (not volatility). Let \(y>0\) be realized variance and \(\hat y>0\) the forecast:

\[
\mathrm{QLIKE}(y,\hat y) = \frac{y}{\hat y} - \log\frac{y}{\hat y} - 1
\]

QLIKE is undefined for non-positive forecasts; those rows are counted as failures, not silently clipped to epsilon unless `qlike_floor` is set in config (default \(10^{-12}\)).

## EWMA variance

\[
\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1-\lambda) r_{t-1}^2
\]

Default \(\lambda=0.94\) (RiskMetrics daily), overridable.

## GARCH-family forecasting contract

The GARCH subsystem is fitted to the causal decimal return series \(r_t\), never to
`future_realized_var_h`. In the current panel trainer, the univariate series is an
ordered, date-level equal-weight mean of finite `ret_1` values across assets on the
gold panel; those assets are PIT-universe members (Wave 108). Training
`panel()` fail-closes if cached gold keys sit outside the current universe
artifact (Wave 109). Panel rows are never
concatenated into a synthetic time series. Per-security GARCH lives in a
separate namespace: `garch_name_forecasts_asof` clones the date-level
`vol_garch.joblib` specification and refits each name on that name's
strictly prior `ret_1` (Wave 112). Those forecasts stamp
`series_scope=security_level_ret_1` and do not replace the market overlay,
`vol_20`, or `max_predicted_vol`. Walk-forward QLIKE/density for that
namespace is `garch_name_walk_forward` (Wave 114). Duplicate
`(security_id, event_time)` keys fail closed. For numerical stability
`arch` receives \(100r_t\), so its conditional variance is in percent-squared units and
is divided by \(10000\) at the API boundary. QLIKE is evaluated on decimal variance,
not sigma.

For symmetric GARCH(\(p,q\)):

\[
\sigma_t^2 = \omega + \sum_{i=1}^{p}\alpha_i\varepsilon_{t-i}^2
              + \sum_{j=1}^{q}\beta_j\sigma_{t-j}^2.
\]

GJR adds \(\sum_i\gamma_i I(\varepsilon_{t-i}<0)\varepsilon_{t-i}^2\), while EGARCH
models log variance and handles positivity in log space. APARCH (Ding–Granger–Engle)
raises the innovation and lagged volatility to a positive power \(\delta\) with
an asymmetry coefficient \(\gamma\in(-1,1)\); FIGARCH (Baillie–Bollerslev–Mikkelsen)
adds a fractional-integration parameter \(d\in(0,1)\) with orders \(p,q\in\{0,1\}\).
The supported innovation families are Gaussian, standardized Student-t, and
skewed Student-t.

`GARCHVol.forecast(horizon=h)` returns a deterministic origin-indexed path of
per-future-bar decimal variances and sigmas. `cumulative_variance` is the sum of
those conditional variances for comparison with the close-to-close forward
realized-variance label. Predictive quantiles and PIT values use the fitted
innovation distribution and are bounded to \([0,1]\) for valid inputs.

The fit gate checks finite parameters, finite positive conditional volatility,
optimizer convergence, and specification-appropriate persistence. For GARCH,
GJR, and APARCH the persistence diagnostic is
\(\sum_i\alpha_i+\sum_j\beta_j\) (GJR adds \(\frac12\sum_i\gamma_i\)); APARCH
also requires \(\delta>0\). FIGARCH requires \(0<d<1\) and is not subjected to
the symmetric GARCH persistence formula. EGARCH is not subjected to that
formula either. Short, constant, non-converged, non-finite, or non-stationary
fits fail closed to a sample-sigma fallback with an explicit diagnostic reason.
`arch` cannot recurse EGARCH or APARCH analytically beyond one step, so
`forecast(horizon>1)` uses a seeded simulation for those specs rather than
repeating the one-step value. This fallback is a safety mechanism, not evidence
of model superiority.

When `vol_realized_garch.joblib` is present, `forecast_asof` and
`optimize_asof` prefer the causal Parkinson Realized GARCH one-step market
variance (Wave 118) with the same fail-closed OHLC / scope / high-frequency-RV
contract as paper/backtest `check_order` (Wave 117). `MarketState.market_risk_overlay`
stamps `realized_garch` versus `garch` so the covariance overlay cannot be
mistaken for return-only GARCH. When that artifact is absent,
`vol_garch.joblib` still supplies the return-only overlay: `forecast_asof`
refits this specification on
the date-level equal-weight `ret_1` series with `event_time` strictly before the
decision origin and attaches the one-step market variance to `MarketState`.
When `available_time` is present, unpublished restatements
(`available_time > asof`) cannot enter that history even if their event time is
earlier; null availability among otherwise usable rows fails closed. When
`silver/universe.parquet` is present the overlay series is restricted to
current PIT members, so paper/backtest execution bars cannot let ineligible
names move the market vol gate. The process-local as-of cache is keyed by a
digest of the full causal history, not the terminal observation alone, and
by the SHA-256 of `vol_garch.joblib` bytes rather than mtime or a partial
spec tuple (Wave 113). The ridge ranker cache is keyed the same way on
`ranker_ridge.joblib` bytes (Wave 121), so an in-place rewrite that keeps
mtime cannot reuse stale alpha.
`optimize_asof` then scales the trailing name-covariance so that the equal-weight
portfolio variance matches that overlay level when `optimizer.covariance` is
the default Ledoit–Wolf path or named `oas` / `ledoit_wolf_nonlinear` / `sample`. Named `optimizer.covariance=dcc_gaussian` uses
Engle (2002) one-step \(H_{t+1}\) instead and does not apply this overlay
(Wave 124). Named `optimizer.covariance=dcc_student_t` uses the Wave 126
Student-t DCC one-step matrix the same way (Wave 127) and must not silently
run Gaussian DCC or Ledoit–Wolf. Named `optimizer.covariance=adcc` uses the
Wave 128 scalar CES ADCC one-step matrix the same way (Wave 129) and must
not silently run Gaussian DCC, Student-t DCC, or Ledoit–Wolf. Named `optimizer.covariance=ewma` uses one-step RiskMetrics \(H_{t+1}\) (Wave 130) and must not silently run DCC or Ledoit–Wolf. Named `optimizer.covariance=oas` uses trailing Chen–Wiesel–Eldar–Hero OAS plus the overlay (Wave 131) and must not silently run Ledoit–Wolf, sample, EWMA, or DCC. Named `optimizer.covariance=ledoit_wolf_nonlinear` uses trailing analytical 2020 nonlinear shrinkage plus the overlay (Wave 140) and must not silently run 2004 linear Ledoit–Wolf, OAS, sample, EWMA, or DCC. Named `optimizer.covariance=sample` uses trailing unbiased sample covariance plus the overlay (Wave 132) and must not silently run Ledoit–Wolf, OAS, EWMA, or DCC. Named `optimizer.covariance=ccc` uses Bollerslev (1990) one-step \(H_{t+1}=D_{t+1} R D_{t+1}\) (Wave 134) and must not silently run Gaussian DCC, Student-t DCC, scalar ADCC, or Ledoit–Wolf. That DCC sample is the trailing contiguous complete-case window
(Wave 125): holes are not concatenated, and an incomplete asof row fails
closed so \(z_t\) cannot be dropped from advertised \(H_{t+1}\). That trailing
matrix is itself
PIT-filtered: when `available_time` is present, unpublished restatements
(`available_time > asof`) cannot enter Ledoit–Wolf, OAS, or sample covariance, and
null availability among usable `ret_1` rows fails closed (Wave 119). The
`/risk/portfolio` research diagnostic uses the same trailing-return helper
and the same overlay scale (Wave 120), stamping `market_risk_overlay` so
unscaled Ledoit–Wolf cannot be mistaken for the optimizer's risk object. A
present Realized GARCH artifact still fail-closes on missing OHLC rather than
reporting unscaled sample risk. Frames without
`available_time` keep the legacy event-time path. Paper and backtest `check_order`
compare `max_predicted_vol` to the same causal one-step Parkinson Realized GARCH
sigma when `vol_realized_garch.joblib` is present (Wave 117–118); otherwise they
use this return-only GARCH overlay when `vol_garch.joblib` is present.
Per-name `vol_20` remains the impact/cost sigma and the fallback when no
artifact or no strictly-prior history exists. The overlay is not a
substitute for `vol_20`. Per-security causal forecasts are a separate
`garch_name_forecasts_asof` namespace with `series_scope=security_level_ret_1`
(Wave 112) and are not wired into `check_order`. A present artifact whose
`series_scope` is not `date_level_equal_weight_cross_section` fails closed.

Walk-forward GARCH QLIKE is scored on that same date-level object: the one
forecast per origin is compared with the equal-weight cross-section of
`future_realized_var_h`, not with concatenated name-level rows. When
`available_time` is present, each origin refits on returns observable at that
origin rather than on a pre-aggregated series that already includes unpublished
restatements. Primary
`qlike` uses a Hansen–Lunde nonoverlapping origin subsample (stride \(h\) on
the session index) so consecutive \(h\)-bar realized windows are not
double-counted; `qlike_overlapping_dates` remains the diagnostic on every
origin. The volatility bench collapses holdout losses to one observation per
date and uses Hansen–Hodrick HAC lags of at least \(h-1\) for Diebold–Mariano.
This is overlap-aware scoring honesty, not a live-performance or per-name
GARCH claim.

The advertised one-step predictive density is scored separately on date-level
equal-weight \(r_t\) at each origin (Wave 105). The fit uses returns with
event time strictly before the origin; the scored realization is that origin's
date-level `ret_1`, never `future_realized_var_h`. Primary `log_score_one_step`
is the Gneiting–Raftery logarithmic score (higher is better) and
`crps_one_step` is a proper CRPS loss; `ignorance_one_step` is the negative
log-score. Gaussian CRPS uses the closed form; t/skew-t CRPS is a quantile
Riemann approximation from the fitted `arch` ppf because GARCH Student-t is
variance-standardized. PIT KS is a calibration diagnostic only (PITs remain
serially dependent under volatility clustering). Companion `*_qlike_origins`
keys reuse the Hansen–Lunde stride so density diagnostics can be compared on
the same origin subset as QLIKE. Missing origin `ret_1` fails closed. This is
not a live-performance or per-name GARCH claim.

Name-level walk-forward QLIKE/density is a separate namespace (Wave 114).
`garch_name_walk_forward` clones the training GARCH specification and refits
each `security_id` on that name's strictly prior `ret_1`. Primary `qlike`
pools the union of per-name Hansen–Lunde nonoverlapping origins; it never
equal-weight collapses names within a date. Date-level `overlap_aware_qlike`
still rejects within-date forecast disagreement, so a name-level forecast
cannot be scored as a market overlay. One-step log-score, CRPS, and PIT KS
use that name's origin `ret_1` (evaluation vintage), never
`future_realized_var_h` and never the cross-section mean. Metrics stamp
`scoring_scope=security_level_ret_1` and do not replace `vol_20`,
`max_predicted_vol`, or date-level `train_volatility` scores. Missing origin
`ret_1`, empty prior history, and duplicate `(security_id, event_time)` keys
fail closed. This is not a live-performance claim.

Log-linear Realized GARCH (Hansen–Huang–Shek 2012) is a separate Gaussian
namespace. The return equation is the usual \(r_t=\mu+\sqrt{h_t}z_t\). Variance
and measurement equations are

\[
\log h_t=\omega+\beta\log h_{t-1}+\gamma\log x_{t-1},
\]

\[
\log x_t=\xi+\varphi\log h_t+\tau_1 z_t+\tau_2(z_t^2-1)+u_t.
\]

The realized measure \(x_t\) is one-day Parkinson variance from daily OHLC,
not intraday RV and never a silent \(r_t^2\) substitute. Persistence is
\(\beta+\gamma\varphi<1\). One-step \(h_{t+1}\) is deterministic given the last
observable \(x_t\); multi-step paths use the plugin \(\mathbb{E}[\log x]=\xi+\varphi\log h\)
and stamp `multi_step_method=expected_log_variance_plugin`.
`realized_garch_market_forecast_asof` clones `vol_realized_garch.joblib` and
refits on date-level equal-weight `ret_1` paired with same-name Parkinson,
with the same PIT `available_time` / universe contract as the return-only
overlay. `forecast_asof` / `optimize_asof` and paper/backtest `check_order`
prefer that one-step Parkinson sigma when the artifact is present (Wave 118);
return-only GARCH remains the fallback overlay, and per-name `vol_20`
remains the impact/cost sigma. A present Realized GARCH artifact whose
`series_scope` is wrong, that claims high-frequency RV, or whose panel
lacks daily OHLC fails closed rather than substituting \(r_t^2\). Missing
OHLC fail closed.

## HAR-RV (Corsi)

\[
\mathrm{RV}_{t+1} = \beta_0 + \beta_d \mathrm{RV}_t + \beta_w \overline{\mathrm{RV}}_{t-4:t} + \beta_m \overline{\mathrm{RV}}_{t-21:t} + \varepsilon_{t+1}
\]

When daily RV is unavailable, close-to-close squared log return proxies RV. Log-RV regression is used when `volatility.har_log: true`.

## Sample covariance

Unbiased Pearson sample covariance (`np.cov(..., ddof=1)`) on the listwise-complete trailing window. The public matrix is trailing (`covariance_object=trailing`, `sample=listwise_complete`), not sequential one-step \(H_{t+1}\). Interior holes are dropped; an incomplete asof row is omitted rather than fail-closed. Params stamp `family=sample`, `spec=unbiased_sample`, and `ddof=1`. `/models` lists `sample` among implemented covariance estimators and named optimizer paths. `optimize_asof` / `/risk/portfolio` use this matrix when `optimizer.covariance=sample` (Wave 132) and apply the GARCH/RGARCH overlay (trailing sample has no \(D_{t+1}\)). The named path must not silently size as Ledoit–Wolf, OAS, EWMA, Gaussian DCC, Student-t DCC, or scalar ADCC. When \(T>N\) the estimator stays sample rather than switching to Ledoit–Wolf. The default Ledoit–Wolf path stays Ledoit–Wolf when \(T\le N\) (Wave 139) rather than silently switching to this unbiased spec. Factor stays unwired. This is not matrix AG-DCC and does not invent high-frequency RV.

## Ledoit–Wolf shrinkage

sklearn `LedoitWolf` shrinks the sample covariance \(S\) toward \(\mu I\):

\[
\Sigma = (1-\delta)S + \delta \mu I, \qquad \mu = \mathrm{tr}(S)/N
\]

This is the 2004 linear formula, not nonlinear spectral shrinkage. Named `optimizer.covariance=ledoit_wolf_nonlinear` is the 2020 analytical formula (Wave 140) and must not silently size as this 2004 path. The public matrix is trailing and listwise-complete (`covariance_object=trailing`, `sample=listwise_complete`), not sequential one-step \(H_{t+1}\). Interior holes are dropped; an incomplete asof row is omitted rather than fail-closed. Params stamp `family=ledoit_wolf`, `spec=ledoit_wolf_2004_linear`, and the fitted shrinkage intensity. `/models` lists `ledoit_wolf` among implemented covariance estimators and the default optimizer path. `optimize_asof` / `/risk/portfolio` use this matrix when `optimizer.covariance=ledoit_wolf` (default) and apply the GARCH/RGARCH overlay (trailing shrinkage has no \(D_{t+1}\)). When \(T\le N\) the estimator stays Ledoit–Wolf (Wave 139) rather than silently switching to unbiased sample. Named `sample` remains the explicit sample path. Generic `shrinkage` stays unknown. This does not invent high-frequency RV or wire factor covariance.

## Analytical nonlinear Ledoit–Wolf shrinkage

Ledoit–Wolf (2020, *Annals of Statistics*) analytical nonlinear shrinkage maps each sample eigenvalue through a kernel estimate of the limiting spectral density and its Hilbert transform (bandwidth \(n^{-1/3}\), \(n\) the effective sample size after demeaning). This is the closed-form successor to QuEST, not numerical QuEST inversion (Ledoit–Wolf 2017 RFS) and not 2004 linear shrinkage toward \(\mu I\). Sample covariance uses \(1/n\) on the centered trailing window, not unbiased `ddof=1`. When \(T\le N\) the singular-case formula is used rather than switching to sample or 2004 linear shrinkage. Effective sample size after demeaning must be at least 12. The public matrix is trailing and listwise-complete (`covariance_object=trailing`, `sample=listwise_complete`), not sequential one-step \(H_{t+1}\). Interior holes are dropped; an incomplete asof row is omitted rather than fail-closed. Params stamp `family=ledoit_wolf_nonlinear`, `spec=ledoit_wolf_2020_analytical`, `demean=true`, `n_eff`, `concentration`, and `bandwidth`. `/models` lists `ledoit_wolf_nonlinear` among implemented covariance estimators and named optimizer paths. `optimize_asof` / `/risk/portfolio` use this matrix when `optimizer.covariance=ledoit_wolf_nonlinear` (Wave 140) and apply the GARCH/RGARCH overlay (trailing shrinkage has no \(D_{t+1}\)). The named path must not silently size as 2004 linear Ledoit–Wolf, OAS, sample, EWMA, or DCC. Generic `shrinkage` / `ledoit_wolf_2017` / `quest` stay unknown so analytical 2020 cannot masquerade as 2004 linear or as QuEST. Default remains trailing Ledoit–Wolf 2004 plus the overlay. Factor stays unwired. This does not invent high-frequency RV.

## Oracle Approximating Shrinkage

sklearn `OAS` (Chen, Wiesel, Eldar, Hero 2010) is a linear shrinkage estimator of the sample covariance toward \(\mu I\). It is not Ledoit–Wolf 2004 and not nonlinear spectral shrinkage. The public matrix is trailing and listwise-complete (`covariance_object=trailing`, `sample=listwise_complete`), not sequential one-step \(H_{t+1}\). Interior holes are dropped; an incomplete asof row is omitted rather than fail-closed. Params stamp `family=oas`, `spec=chen_wiesel_eldar_hero_2010`, and the fitted shrinkage intensity. `/models` lists `oas` among implemented covariance estimators and named optimizer paths. `optimize_asof` / `/risk/portfolio` use this matrix when `optimizer.covariance=oas` (Wave 131) and apply the GARCH/RGARCH overlay (trailing shrinkage has no \(D_{t+1}\)). The named path must not silently size as Ledoit–Wolf, nonlinear Ledoit–Wolf, sample, EWMA, Gaussian DCC, Student-t DCC, or scalar ADCC. When \(T\le N\) the estimator stays OAS rather than switching to sample. Generic `shrinkage` stays unknown. Default remains trailing Ledoit–Wolf plus the overlay. This is not matrix AG-DCC and does not invent high-frequency RV.

## RiskMetrics EWMA covariance

One-step RiskMetrics recursion on decimal returns:

\[
H_{t+1}=\lambda H_t+(1-\lambda)r_t r_t'
\]

with \(\lambda\in[0,1)\) from `features.ewma_lambda` (default \(0.94\)). The public matrix is \(H_{t+1}\) that includes asof \(r_t\), not in-sample last \(H_t\) that re-applies \(r_{t-1}\) and drops \(r_t\). The estimation sample is the trailing contiguous complete-case window ending at the last row (same sequential honesty as DCC Wave 125): listwise deletion must not concatenate non-adjacent days, and an incomplete terminal row fails closed. Params stamp `family=ewma`, `spec=jpmorgan_riskmetrics_1996`, `covariance_object=one_step_ahead`, `horizon=1`, and `sample=trailing_complete_window`. `/models` lists `ewma` among implemented covariance estimators and named optimizer paths. `optimize_asof` / `/risk/portfolio` use this matrix when `optimizer.covariance=ewma` (Wave 130) and must not overlay GARCH/RGARCH (the EWMA recursion already supplies \(H_{t+1}\)) or silently size as sample, Ledoit–Wolf, OAS, Gaussian DCC, Student-t DCC, or scalar ADCC. Generic `dcc` stays unknown. Default remains trailing Ledoit–Wolf plus the overlay. This is not matrix AG-DCC and does not invent high-frequency RV.

## CCC (Bollerslev 1990)

Two-stage Constant Conditional Correlation. Stage 1 is univariate Gaussian
GARCH(1,1) via `arch` / `GARCHVol`, the same stage-1 contract as Gaussian DCC.
Stage 2 is the constant correlation of standardized residuals:

\[
R=\mathrm{corr}(z),\qquad H_{t+1}=D_{t+1} R D_{t+1}.
\]

There is no \(Q\) recursion and no \(a,b\) QML: this is not Gaussian DCC with
\(a=b=0\) fitted by Engle QML. `ccc` does not call `dcc_gaussian`,
`dcc_student_t`, or `adcc`. \(D_{t+1}\) is the univariate GARCH one-step
sigma, not in-sample last \(\sigma_t\). The estimation sample is the trailing
contiguous complete-case window. Params stamp `family=ccc`,
`spec=bollerslev_1990_ccc`, `dist=normal`, `asymmetric=false`,
`dynamic_correlation=false`, `covariance_object=one_step_ahead`, `horizon=1`,
and `sample=trailing_complete_window`. `/models` lists it among implemented
covariance estimators and named optimizer paths. `optimize_asof` /
`/risk/portfolio` use this matrix when `optimizer.covariance=ccc` (Wave 134)
and must not overlay GARCH/RGARCH (CCC already supplies \(D_{t+1}\)) or
silently size as Gaussian DCC, Student-t DCC, scalar ADCC, or Ledoit–Wolf.
Generic `dcc` stays unknown. Named `agdcc` is diagonal CES AG-DCC
(Wave 136); named `agdcc_full` is unrestricted CES AG-DCC (Wave 138)
and must not silently size as diagonal AG-DCC. This does not invent
high-frequency RV.

## DCC(1,1)

Two-stage Engle (2002). Stage 1: univariate Gaussian GARCH(1,1) via `arch` /
`GARCHVol` on each name's decimal returns (Wave 116). Standardized residuals
\(z_t\) are the fitted GARCH innovations, not RiskMetrics EWMA. A failed,
short, zero-variance, or fallback univariate fit fails closed rather than
silently substituting EWMA residuals. Stage 2:

\[
Q_t = (1-a-b)\bar Q + a z_{t-1}z_{t-1}^\top + b Q_{t-1}
\]

\[
R_t = \mathrm{diag}(Q_t)^{-1/2} Q_t \mathrm{diag}(Q_t)^{-1/2}, \qquad H_t = D_t R_t D_t
\]

Parameters \(a,b>0\), \(a+b<1\) are estimated by QML on the correlation likelihood. Implemented in-house (ADR-004). The public matrix is the one-step-ahead Engle object \(H_{t+1}=D_{t+1}R_{t+1}D_{t+1}\) (Wave 123): \(Q_{t+1}\) uses the last standardized residual \(z_t\), and \(D_{t+1}\) is the univariate GARCH one-step sigma, not in-sample last \(\sigma_t\). The estimation sample is the trailing contiguous complete-case window ending at the last row (Wave 125): listwise deletion must not concatenate non-adjacent days, and an incomplete terminal row fails closed so advertised \(H_{t+1}\) cannot drop asof \(z_t\). Params stamp `family=dcc_gaussian`, `dist=normal`, `asymmetric=false`, `covariance_object=one_step_ahead`, `horizon=1`, and `sample=trailing_complete_window`. Student-t DCC (Wave 126) uses the same Engle \(Q\) recursion, one-step \(H_{t+1}\), and trailing complete window. Stage 1 is univariate Student-t GARCH(1,1). Stage 2 QML uses the covariance Student-t correlation likelihood with a single \(ν>2\) (scale \(((ν-2)/ν)R\) so \(\mathrm{Var}(z)=R\)). Params stamp `family=dcc_student_t`, `dist=student_t`, `spec=engle_2002_student_t_dcc`, and `nu`. \(ν\le 2\) fails closed. Scalar Cappiello–Engle–Sheppard ADCC is a separate catalog estimator (Wave 128); `adcc` must not silently run Gaussian or Student-t DCC. `optimize_asof` / `/risk/portfolio` use Gaussian DCC when `optimizer.covariance=dcc_gaussian` (Wave 124), Student-t DCC when `optimizer.covariance=dcc_student_t` (Wave 127), scalar CES ADCC when `optimizer.covariance=adcc` (Wave 129), RiskMetrics EWMA when `optimizer.covariance=ewma` (Wave 130), Bollerslev CCC when `optimizer.covariance=ccc` (Wave 134), and diagonal CES AG-DCC when `optimizer.covariance=agdcc` (Wave 136), and unrestricted CES AG-DCC when `optimizer.covariance=agdcc_full` (Wave 138); those named paths do not apply the GARCH/RGARCH overlay to \(H_{t+1}\) and must not silently substitute for each other or for Ledoit–Wolf. Named `optimizer.covariance=oas` is trailing Chen OAS plus the overlay (Wave 131), not a one-step DCC/EWMA path. Named `optimizer.covariance=sample` is trailing unbiased sample covariance plus the overlay (Wave 132) and must not silently size as Ledoit–Wolf, OAS, EWMA, or DCC. Ambiguous aliases (`gaussian`, `normal`, `t`, `student_t`) are rejected. Generic `dcc` stays unknown. The default remains trailing Ledoit–Wolf plus the overlay. Bollerslev (1990) CCC is a separate catalog estimator (Wave 133) and named optimizer path (Wave 134); `ccc` must not silently run Gaussian DCC, Student-t DCC, or ADCC. Diagonal CES AG-DCC is a separate catalog estimator (Wave 135) and named optimizer path (Wave 136); `agdcc` must not silently run scalar ADCC, Gaussian DCC, CCC, unrestricted AG-DCC, or Ledoit–Wolf. Unrestricted full-matrix AG-DCC is a catalog estimator (Wave 137) and named optimizer path (Wave 138); `agdcc_full` must not silently run diagonal AG-DCC, scalar ADCC, Gaussian DCC, CCC, or Ledoit–Wolf. Factor stays unwired.

## Scalar ADCC (Cappiello–Engle–Sheppard 2006)

Two-stage scalar ADCC. Stage 1 is univariate Gaussian GARCH(1,1) via `arch` / `GARCHVol`, the same stage-1 contract as Gaussian DCC. Stage 2 QML estimates \(a,b,g\) on

\[
n_t = I[z_t < 0]\odot z_t, \qquad \bar N = \mathbb{E}[n_t n_t^\top]
\]

\[
Q_t = (1-a-b)\bar Q - g\bar N + a z_{t-1}z_{t-1}^\top + b Q_{t-1} + g n_{t-1}n_{t-1}^\top
\]

with \(R_t\) and \(H_t\) as in Engle DCC. Positive-definiteness uses \(a,b,g\ge 0\) and \(a+b+\kappa g<1\), where \(\kappa=\lambda_{\max}(\bar Q^{-1/2}\bar N\bar Q^{-1/2})\). This is not Gaussian DCC with an `asymmetric` stamp: `adcc` does not call `dcc_gaussian`. The public matrix is one-step \(H_{t+1}=D_{t+1}R_{t+1}D_{t+1}\) on the trailing contiguous complete-case window. Params stamp `family=adcc`, `spec=cappiello_engle_sheppard_2006`, `dist=normal`, `asymmetric=true`, `g`, and `kappa`. `/models` lists it among implemented covariance estimators and named optimizer paths. `optimize_asof` / `/risk/portfolio` use this matrix when `optimizer.covariance=adcc` (Wave 129) and must not silently size as Gaussian DCC, Student-t DCC, CCC, diagonal AG-DCC, or Ledoit–Wolf. This is a scalar CES correlation spec, not diagonal AG-DCC and not high-frequency RV.

## Diagonal AG-DCC (Cappiello–Engle–Sheppard 2006)

Two-stage diagonal AG-DCC. Stage 1 is univariate Gaussian GARCH(1,1) via `arch` / `GARCHVol`. Stage 2 QML estimates diagonal \(A=\mathrm{diag}(a)\), \(B=\mathrm{diag}(b)\), \(G=\mathrm{diag}(g)\) on

\[
Q_t = (\bar Q - A\bar Q A - B\bar Q B - G\bar N G) + A z_{t-1}z_{t-1}^\top A + B Q_{t-1} B + G n_{t-1}n_{t-1}^\top G
\]

with \(n_t=I[z_t<0]\odot z_t\) and \(\bar N=\mathbb{E}[n_t n_t^\top]\) as in scalar ADCC. Equal diagonals \(A=\sqrt{a}I\), \(B=\sqrt{b}I\), \(G=\sqrt{g}I\) recover the scalar CES recursion; heterogeneous diagonals do not. This is not scalar ADCC with a `parameterization` stamp: `agdcc` does not call `adcc`. Positive-definiteness uses \(a,b,g\ge 0\) and a positive-definite intercept. The public matrix is one-step \(H_{t+1}=D_{t+1}R_{t+1}D_{t+1}\) on the trailing contiguous complete-case window. Params stamp `family=agdcc`, `spec=cappiello_engle_sheppard_2006_diagonal_agdcc`, `parameterization=diagonal`, `dist=normal`, and `asymmetric=true`. `/models` lists `agdcc` among implemented covariance estimators and named optimizer paths. `optimize_asof` / `/risk/portfolio` use this matrix when `optimizer.covariance=agdcc` (Wave 136) and must not overlay GARCH/RGARCH (AG-DCC already supplies \(D_{t+1}\)) or silently size as scalar ADCC, Gaussian DCC, CCC, unrestricted AG-DCC, or Ledoit–Wolf. Unrestricted full-matrix AG-DCC (`agdcc_full`) is a separate catalog estimator (Wave 137) and named optimizer path (Wave 138). This does not invent high-frequency RV or wire factor covariance.

## Unrestricted AG-DCC (Cappiello–Engle–Sheppard 2006)

Two-stage unrestricted AG-DCC. Stage 1 is univariate Gaussian GARCH(1,1) via `arch` / `GARCHVol`. Stage 2 QML estimates unrestricted matrices \(A,B,G\) on

\[
Q_t = (\bar Q - A\bar Q A^\top - B\bar Q B^\top - G\bar N G^\top) + A z_{t-1}z_{t-1}^\top A^\top + B Q_{t-1} B^\top + G n_{t-1}n_{t-1}^\top G^\top
\]

with \(n_t=I[z_t<0]\odot z_t\) and \(\bar N=\mathbb{E}[n_t n_t^\top]\) as in scalar ADCC. Diagonal \(A,B,G\) recover Wave 135 diagonal AG-DCC; nonzero off-diagonals do not. This is not diagonal AG-DCC with a `parameterization` stamp: `agdcc_full` does not call `agdcc`, `adcc`, `dcc_gaussian`, `dcc_student_t`, or `ccc`. Positive-definiteness uses a positive-definite intercept and spectral radius of \(A\otimes A+B\otimes B+G\otimes G\) strictly below one. The public matrix is one-step \(H_{t+1}=D_{t+1}R_{t+1}D_{t+1}\) on the trailing contiguous complete-case window. Params stamp `family=agdcc_full`, `spec=cappiello_engle_sheppard_2006_full_agdcc`, `parameterization=full`, `dist=normal`, and `asymmetric=true`. `/models` lists `agdcc_full` among implemented covariance estimators and named optimizer paths. `optimize_asof` / `/risk/portfolio` use this matrix when `optimizer.covariance=agdcc_full` (Wave 138) and must not overlay GARCH/RGARCH (AG-DCC already supplies \(D_{t+1}\)) or silently size as diagonal AG-DCC, scalar ADCC, Gaussian DCC, CCC, or Ledoit–Wolf. This does not invent high-frequency RV or wire factor covariance.

## PSD repair

If \(\min\mathrm{eig}(\Sigma) < -\varepsilon\) or \(\Sigma \ne \Sigma^\top\) beyond tolerance, repair with Higham (1988) eigenvalue clipping (statsmodels `cov_nearest`, method `clipped`) and log the Frobenius change. The optimizer never receives an unrepaired indefinite matrix.

## VaR and Expected Shortfall

**Loss convention:** \(L = -R\) for a return \(R\). VaR at level \(\alpha\) (e.g. 0.95) is the \(\alpha\)-quantile of \(L\):

\[
\mathrm{VaR}_\alpha = q_\alpha(L)
\]

Expected Shortfall (Acerbi–Tasche):

\[
\mathrm{ES}_\alpha = \mathbb{E}[L \mid L \ge \mathrm{VaR}_\alpha]
\]

For discrete samples, ES is the mean of losses strictly exceeding VaR plus a boundary correction when ties sit on VaR. Sign is always **loss-positive**: a 5% left-tail return of \(-3\%\) is VaR \(=0.03\).

## Maximum drawdown

Wealth \(W_t = \prod_{u\le t}(1+R_u)\), peak \(\mathrm{Peak}_t = \max_{u\le t} W_u\):

\[
\mathrm{DD}_t = W_t/\mathrm{Peak}_t - 1, \qquad \mathrm{MDD} = \min_t \mathrm{DD}_t
\]

MDD is \(\le 0\). Monotone increasing wealth implies MDD \(= 0\). If any return or drawdown observation is non-finite, the metric is NaN rather than a minimum over a silently truncated finite subset.

## Sharpe ratio

For periodic excess returns with \(n\) observations per year (default 252):

\[
\mathrm{SR} = \frac{\overline{r}_{\mathrm{ex}}}{\mathrm{std}(r_{\mathrm{ex}})}\sqrt{n}
\]

Annualization is skipped and a warning is emitted if timestamps are irregular. Sample count is always reported. Suspiciously large \(|\mathrm{SR}|>5\) is flagged, not suppressed.

## Probabilistic and Deflated Sharpe

Bailey & López de Prado. PSR is \(P(\mathrm{true\,SR} > \mathrm{SR}^*)\) under a non-Normal standard error (Lo 2002):

\[
\widehat{\mathrm{SE}} = \sqrt{1 - \gamma_3\,\mathrm{SR} + \frac{\gamma_4-1}{4}\mathrm{SR}^2}
\]

\[
\mathrm{PSR} = \Phi\Bigl((\mathrm{SR}-\mathrm{SR}^*)\sqrt{T-1}/\widehat{\mathrm{SE}}\Bigr)
\]

DSR uses \(\mathrm{SR}^* = \mathbb{E}[\max \widehat{\mathrm{SR}}]\) under \(N\) trials (Euler–Mascheroni \(\gamma\approx 0.57721\)):

\[
\mathrm{SR}^* = \sqrt{V[\widehat{\mathrm{SR}}]}\Bigl[(1-\gamma)\Phi^{-1}(1-1/N)+\gamma\Phi^{-1}(1-1/(Ne))\Bigr]
\]

Kurtosis \(\gamma_4\) is **raw** (not excess). \(N\) is the recorded trial count, including failed experiments.

## Information coefficient

\[
\mathrm{IC}_t = \mathrm{Corr}(\hat s_{i,t},\, y_{i,t+h}), \qquad
\mathrm{RankIC}_t = \mathrm{Spearman}(\hat s_{i,t},\, y_{i,t+h})
\]

\[
\mathrm{ICIR} = \frac{\overline{\mathrm{IC}}}{\mathrm{std}(\mathrm{IC})}
\]

ICIR is **not** annualized unless explicitly reported as `icir_ann = ICIR * sqrt(252)`. Ranks are computed within timestamp, never across dates. Inference on \(\overline{\mathrm{IC}}\) uses Newey–West HAC standard errors on the date-level IC series.

## Newey–West HAC

For a scalar series \(x_t\) with mean \(\bar x\):

\[
\gamma_j = \frac{1}{n}\sum_{t=j+1}^{n}(x_t-\bar x)(x_{t-j}-\bar x),\qquad
\Omega=\gamma_0+2\sum_{j=1}^{L}\Big(1-\frac{j}{L+1}\Big)\gamma_j
\]

\(\mathrm{Var}(\bar x)=\Omega/n\). Default lag \(L=\lfloor 1.5 n^{1/3}\rfloor\).

## Diebold–Mariano

Let \(d_t=L(e_{A,t})-L(e_{B,t})\). Test \(E[d_t]=0\) with HAC t-stat. Negative mean loss differential prefers model A. Smaller loss is better.

## Benjamini–Hochberg FDR

For \(m\) tests, reject \(p_{(k)}\le \alpha k/m\) up to the largest such \(k\).

BH is applied **within** two hypothesis families, never pooled: **calibration** (Kupiec/Ville; fail-to-reject is success) and **discovery** (signal/contrast tests; reject is a finding). Jackknife+ H10 is a **floor check** against \(1-2\alpha\), not a Kupiec null, and is not FDR-adjusted.

## Data-snooping (Reality Check / SPA / StepM / MCS)

Universe of \(K\) strategies with period-level performance differentials
\(f_{t,k}\) (strategy minus benchmark; **larger is better**). All tests resample
time with the **stationary bootstrap** (Politis–Romano 1994): geometric block
lengths with mean \(L\), a new block starting with probability \(1/L\) at a
uniform index (wrapping). Default \(L\) is the **Politis–White (2004)**
automatic block length for the mean (flat-top kernel,
\(K_N=\max(10,\lceil\sqrt n\rceil)\), \(\hat m\) the smallest \(m\) with
\(|\hat\rho(m+k)|<2\sqrt{\log_{10}n/n}\),
\(\hat b=(2\hat G^2/\hat D^2)^{1/3}n^{1/3}\) clamped to \([1,n-1]\)).

- **Reality Check** (White 2000): \(V=\max_k\sqrt T\,\bar f_k\); the bootstrap
  null recenters every column at its sample mean,
  \(V^*_b=\max_k\sqrt T(\bar f^*_{k,b}-\bar f_k)\). One-sided
  \(p=(1+\#\{V^*\ge V\})/(B+1)\) for \(H_0:\max_k E[f_k]\le 0\).
- **SPA** (Hansen 2005): studentized \(T^{SPA}=\max_k\sqrt T\bar f_k/\hat\sigma_k\),
  with \(\hat\sigma_k\) the bootstrap long-run sd of \(\sqrt T\bar f_k\). Null
  bootstrap \(T^*_b=\max_k\sqrt T(\bar f^*_{k,b}-g_k)/\hat\sigma_k\) under three
  recentering rules: \(g_k=\bar f_k\) (**lower**), \(g_k=0\) (**upper**,
  conservative), and
  \(g_k=\bar f_k\,\mathbf 1\{\sqrt T\bar f_k/\hat\sigma_k\ge-\sqrt{2\log\log T}\}\)
  (**consistent**, recommended). There is no universal ordering between the
  three: consistent equals lower exactly when no column is significantly
  negative, and can sit below lower when bad columns inflate the all-recentered
  null.
- **StepM** (Romano–Wolf 2005): order by ascending studentized statistic; at
  each step the null distribution is the bootstrap maximum over the remaining
  (less significant) hypotheses; adjusted p-values are the cumulative maximum
  taken from the most significant end, so rejections are exactly
  \(\{p^{adj}\le\alpha\}\) and FWER is controlled.
- **MCS** (Hansen–Lunde–Nason 2011): within the active set, demean
  cross-sectionally (H0: equal means), \(T_{max}=\max_i t_i-\min_i t_i\) on
  studentized demeaned means; while the bootstrap p-value is below \(\alpha\),
  eliminate the worst (smallest demeaned mean). The MCS is
  \(\{i: p_i\ge\alpha\}\).

Fail-closed: non-finite rows are dropped with an explicit count; zero-variance
columns are dropped (studentization is undefined); panels shorter than 10
observations return honest NaN results; shape violations raise `ValueError`.
The ranker universe battery is stamped as `families["ranking"]["data_snooping"]`
and minted as **H99_ranking_data_snooping** (discovery) when the consistent SPA
p-value is finite. Research diagnostic only — never a live Sharpe / promotion
claim.

## CSCV / PBO

Combinatorial purged CV (López de Prado). PBO is the fraction of CSCV splits in which the in-sample-best trial has below-median out-of-sample performance (strict `<` median; ties do not count as overfitting).

Implementation note: Dipcatcher's `combinatorial_purged_cv` applies purge+embargo
**per contiguous test group** (not the union span of all test groups in a fold),
so intervening train between non-adjacent test blocks is preserved. See
`docs/VALIDATION.md` § CPCV and Wave 25. Invalid CPCV group counts / empty groups
raise ``ValueError``; aggressive purge/embargo may honestly yield fewer folds than
``C(n_groups, n_test_groups)`` (including zero). ``probability_of_backtest_overfitting``
refuses empty / mismatched / too-small / non-finite IS–OOS matrices with NaN
(never a silent PBO of 0.0 on invalid input). Research-diagnostic only.

## Mean-variance objective

\[
\max_w \; w^\top \alpha - \lambda_{\mathrm{risk}}\, w^\top \Sigma w - \lambda_{\mathrm{tc}}\,\mathrm{TC}(w-w_{\mathrm{prev}}) - \lambda_{\mathrm{tail}}\,\mathrm{Tail}(w)
\]

subject to configurable gross, net, name, sector, factor, turnover, and ADV constraints. Aggregate sleeve caps `long_max` / `short_max` default so `long_max + short_max = gross_leverage`; if sleeves are set tighter, they bind first (`constraints.effective_gross_cap`). Transaction cost \(\mathrm{TC}\) is a convex approximation (L1 spread/fees plus quadratic or \(\ell_2\) square-root linearization; see execution spec). Infeasible problems return diagnostics; constraints are never silently dropped.

## CVaR (Rockafellar–Uryasev)

For scenario losses \(L_s(w)\) and level \(\alpha\):

\[
\mathrm{CVaR}_\alpha = \zeta + \frac{1}{(1-\alpha)N}\sum_s u_s, \qquad u_s \ge L_s(w)-\zeta,\; u_s\ge 0
\]

Modes: minimize CVaR, or maximize \(w^\top\alpha\) subject to \(\mathrm{CVaR}_\alpha \le\) limit.

## Square-root impact

\[
\mathrm{Impact} \approx \Upsilon\,\sigma\,\sqrt{|Q|/\mathrm{ADV}}
\]

\(\Upsilon\) is configured, not claimed universal. Zero quantity implies zero impact.

## Almgren–Chriss

Minimize expected implementation shortfall plus \(\lambda\) times its variance, with temporary impact \(\eta v\) and permanent impact \(\gamma x\). As \(\lambda\to 0\), the schedule approaches equal slices (less urgency). As \(\lambda\) increases, the schedule becomes more front-loaded. Zero parent quantity implies zero cost.

## Component risk

\[
V = w^\top\Sigma w, \quad \sigma_p=\sqrt{V}, \quad
\mathrm{MCR} = \Sigma w / \sigma_p, \quad
\mathrm{CR}_i = w_i (\Sigma w)_i / \sigma_p
\]

Component contributions sum to \(\sigma_p\) up to numerical tolerance.

## Amihud illiquidity

\[
\mathrm{ILLIQ}_i = \mathrm{mean}_t \frac{|R_{i,t}|}{\mathrm{DollarVolume}_{i,t}}
\]

over a configured lookback, skipping zero-volume bars.

## Conformal prediction (CQR + ACI)

Split conformalized quantile regression (Romano, Patterson, Candès 2019) at miscoverage \(\alpha\):

\[
s_i=\max\bigl(q_{\mathrm{lo}}(x_i)-y_i,\, y_i-q_{\mathrm{hi}}(x_i)\bigr),\qquad
\hat q=\mathrm{Quantile}_{((n+1)(1-\alpha))/n}(s)
\]

\[
C(x)=\bigl[q_{\mathrm{lo}}(x)-\hat q,\; q_{\mathrm{hi}}(x)+\hat q\bigr]
\]

Calibration scores are never computed on the test window. Adaptive conformal inference (Gibbs & Candès 2021) updates the miscoverage once per timestamp:

\[
\alpha_{t+1}=\mathrm{clip}\bigl(\alpha_t+\gamma(\alpha-\mathrm{err}_t),\,\varepsilon,1-\varepsilon\bigr)
\]

where \(\mathrm{err}_t\) is the date-level miss rate. After a miss, \(\alpha_t\) falls and sets widen. Coverage and mean/median width are the lab scores. PIT of the raw quantile model is still reported and is not “fixed” by conformalization.

One-sided conformal bounds use \(s_i=y_i-q(x_i)\) and \(C=(-\infty, q+\hat q]\).

Normalized CQR divides residual scores by a PIT-safe scale \(\sigma(x)\) (asset `vol_20`, else predicted interval width):

\[
s_i=\max\bigl((q_{\mathrm{lo}}-y_i)/\sigma_i,\,(y_i-q_{\mathrm{hi}})/\sigma_i\bigr),\qquad
C(x)=\bigl[q_{\mathrm{lo}}-\hat q\,\sigma,\; q_{\mathrm{hi}}+\hat q\,\sigma\bigr]
\]

Mondrian conformal (Vovk) computes a separate \(\hat q_g\) and ACI \(\alpha_t^{(g)}\) inside each calibration tercile of \(\sigma(x)\). Cuts are frozen on the calibration window. This targets coverage conditional on \(X\), not on \(|Y|\). Coverage given large \(|Y|\) is a selected slice of the target and is not a finite-sample conformal guarantee.

The **operational wrappee** is a PIT-safe scaled location-scale on asset `vol_20`: scaled Student-\(t\) \(q_\tau=\mu+\hat\sigma_z\,\sigma(x)\,t_{\nu,\tau}\) ( \(\nu\) clamped to \([3,30]\) ) when it improves holdout CRPS or PIT KS versus scaled Gaussian without undercovering; otherwise scaled Gaussian \(q_\tau=\mu+\hat\sigma_z\,\sigma(x)\,z_\tau\). CRC, one-sided tail conformal, and e-value ACI wrap that same wrappee. Homoskedastic Gaussian remains a named diagnostic (`gaussian_raw` / unscaled VaR) and is never the operational wrappee when vol exists. Raw Gaussian CQR/ACI (`cqr_raw` / `aci_raw`) is the misspecification-repair diagnostic: that is where conformal expands undercovering bands toward \(1-\alpha\). Operational H7/H8 CQR/ACI wrap already-wide scaled bands and may look like the split-conformal identity; they are not evidence that conformal “repaired” a misspecified model. PIT of the quantile model is still reported and is not rewritten by conformalization. Coverage given large \(|Y|\) is not a conformal guarantee. Distribution family scores 1d and 5d as separate keys. Jackknife+ is scored against the paper’s \(1-2\alpha\) bound, not \(1-\alpha\). Vol-scaled historical VaR uses the same \(\sigma(x)\) on losses.

## SOTA lab extensions

The conformal identities above are unchanged. These lab families wrap them; see the ADRs for the accepted option and the coverage/risk identity.

- **E-values (ADR-010).** Bernoulli betting e-process on prediction-set misses versus nominal \(\alpha\). \(E_t=\prod_{s\le t}e_s\) is a nonnegative martingale under the null; Ville's inequality uses threshold \(1/0.05=20\) with no peeking correction. **Loss-diff e-process (Day Wave 4):** Choe–Ramdas-style capital on \(d=L_A-L_B\) for \(H_0:\mathbb{E}[d]\le 0\) — see section above; research-only.
- **Jackknife+ (ADR-011).** Leave-one-out residual conformal wrapper. Finite-sample coverage is \(\ge 1-2\alpha\), not \(1-\alpha\). **Day Wave 41 honesty:** that floor (and H10) is **marginal under exchangeability** (Barber–Candès–Ramdas–Tibshirani 2021), **not** training-conditional — Bian & Barber (2023) show training-conditional coverage can fail without stability. Lab surfaces `coverage_guarantee_scope=marginal_exchangeable` on model meta / `bench_jackknife_plus`; do not read H10 as a training-conditional bound.
- **Conformal Risk Control (ADR-012).** Smallest expansion \(\hat\lambda\) such that \((n\widehat L_n(\lambda)+B)/(n+1)\le\alpha\) for monotone 0-1 VaR-hit loss.
- **Weighted split CQR (ADR-013).** Tibshirani–Barber–Candès–Ramdas likelihood-ratio weights on a PIT-safe vol covariate. Lab scores remain coverage and width.
- **Interval position caps (ADR-014).** \(\mathrm{cap}=\bar w\cdot w_{\mathrm{ref}}/(w_{\mathrm{ref}}+\mathrm{width})\cdot d_{\mathrm{ref}}/(d_{\mathrm{ref}}+\max(0,-\ell))\). Binding fraction and mean cap only; no Sharpe.
- **Quantile Thompson (ADR-015).** Shared linear pinball models on a \(\tau\)-grid; posterior draw then top-\(k\). Reward is the scientific residual, not a portfolio path.
- **CV+ / JAW (ADR-016).** Date-grouped folds; minmax retains the \(1-\alpha\) floor, while plus and JAW are \(1-2\alpha\). **Day Wave 41 honesty:** H15 / `coverage_floor` inherit the same **marginal** exchangeable scope (not training-conditional); `coverage_guarantee_scope=marginal_exchangeable` on model meta / `bench_cv_plus`.
- **Localized conformal (ADR-017).** RBF-weighted CQR on the PIT-safe volatility covariate; report coverage and width.
- **Conformal rank sets (ADR-018).** Date-grouped top-k selection reports set size, FDR, and oracle hit only.
- **Online CRC (ADR-019).** One monotone-loss threshold update per date; cross-sectional names are not iid time observations.
- **Portfolio conformal (ADR-020).** One conformal set per date for the scalar book return \(w^\top r\).
- **Scaled empirical residual density (research comparator).** A non-parametric
  empirical distribution of standardized residuals \(z=(y-\mu)/s(x)\), scaled
  back by the PIT-safe covariate. It is scored by CRPS, PIT KS, and coverage;
  it is not an operational wrappee or a conditional-coverage guarantee.
- **Panel calibration inference (Day Wave 82).** H16–H18 prefer HAC tests on
  date-aggregated miss rates when available; name-level Kupiec remains a
  compatibility diagnostic, not the primary evidence for modern panel runs.

## Kupiec and Christoffersen VaR hit tests

Hits \(I_t = 1\{L_t > \mathrm{VaR}_t\}\). Expected hit rate under a calibrated \(\alpha\)-VaR is \(p = 1-\alpha\).

**Kupiec POF** (unconditional coverage):

\[
LR_{uc} = -2\bigl[(n-x)\log(1-p)+x\log p - (n-x)\log(1-\hat\pi)-x\log\hat\pi\bigr],\quad \hat\pi=x/n
\]

\(LR_{uc}\sim\chi^2_1\) under the null.

**Christoffersen independence**: first-order Markov transitions \(n_{ij}\). Null \(\pi_{01}=\pi_{11}=\hat\pi\). \(LR_{ind}\sim\chi^2_1\).

**Conditional coverage**: \(LR_{cc}=LR_{uc}+LR_{ind}\sim\chi^2_2\).

Implemented in `metrics.probability` (`kupiec_pof`, `christoffersen_independence`, `christoffersen_cc`) and wired through `metrics.analytics.var_backtest_hooks` **and** research `bench_tail` (Day Wave 16: `christoffersen_ind_*` / `christoffersen_cc_*` primary + `*_unscaled` / `*_scaled` mirrors beside Kupiec on the same holdout hit series; miss level \(p=0.05\); empty/short → honest NaN). Day Wave 17: H-table `H4b_var_christoffersen_cc` mints from finite `christoffersen_cc_p` only (calibration; skip-on-nonfinite; no ind hyp). Day Wave 26: soft verify requires that finite `christoffersen_cc_p` notebooks contain `H4b_var_christoffersen_cc` (calibration); `hypothesis_h4b_missing_despite_finite_christoffersen_cc_p` on receipts. Day Wave 29: soft verify requires that finite `kupiec_p` notebooks contain `H4_var_kupiec` (calibration); `hypothesis_h4_missing_despite_finite_kupiec_p` on receipts. Day Wave 30: soft verify requires that finite `volatility.dm_p` notebooks contain `H3_vol_dm` (discovery); `hypothesis_h3_missing_despite_finite_dm_p` on receipts. Day Wave 31: soft verify requires that finite `oracle_raw.p_ic` / `ls_p` in `notebook.rankers` mint `H1_ranking_oracle` / `H2_decile_mono` (discovery; independent gates); `hypothesis_h1_missing_despite_finite_p_ic` / `hypothesis_h2_missing_despite_finite_ls_p` on receipts. Day Wave 32: soft verify requires that finite `conformal.aci.kupiec_p` notebooks contain `H7_aci_coverage` (calibration); `hypothesis_h7_missing_despite_finite_aci_kupiec_p` on receipts. Day Wave 33: soft verify requires that finite `conformal.mondrian_aci.high_x_kupiec_p` notebooks contain `H8_mondrian_high_vol` (calibration); `hypothesis_h8_missing_despite_finite_high_x_kupiec_p` on receipts. Day Wave 34: soft verify requires that finite `crc.kupiec_p` notebooks contain `H11_crc_var` (calibration); `hypothesis_h11_missing_despite_finite_crc_kupiec_p` on receipts. Day Wave 35: soft verify requires that finite `weighted_conformal.kupiec_p` notebooks contain `H12_weighted_cqr` (calibration); `hypothesis_h12_missing_despite_finite_wcqr_kupiec_p` on receipts. Day Wave 36: soft verify requires that finite `evalues.e_sup` notebooks contain `H9_eprocess_aci` (calibration); `hypothesis_h9_missing_despite_finite_e_sup` on receipts. Day Wave 37: soft verify requires that finite `jackknife_plus.coverage` notebooks contain `H10_jackknife_coverage` (bound); `hypothesis_h10_missing_despite_finite_coverage` on receipts. Day Wave 38: soft verify requires that finite `cv_plus.coverage` + `coverage_floor` notebooks contain `H15_cv_plus_floor` (bound); `hypothesis_h15_missing_despite_finite_coverage_and_floor` on receipts. Day Wave 39: soft verify requires that finite panel `localized_conformal`/`online_crc`/`portfolio_conformal` `kupiec_p` (non-fixture) notebooks contain `H16_localized_cqr`/`H17_online_crc`/`H18_portfolio_conformal` (calibration); `hypothesis_h16|h17|h18_missing_despite_finite_kupiec_p` on receipts. Day Wave 40: soft verify requires that finite `conformal_rank.fdr` (non-fixture) notebooks contain `H19_conformal_rank` (bound); `hypothesis_h19_missing_despite_finite_fdr` on receipts. Day Wave 18: soft VaR-battery verify requires Christoffersen CC+ind key presence whenever nonempty `tail` exposes `kupiec_p` or `kupiec_lr` (NaN values ok; empty `{}` skips; `tail_var_battery_incomplete:<key>` on receipts; scorecard `tail_var_battery_ok`). Research-diagnostic only — never live capital or promotion evidence. Day Wave 21: soft ES-battery verify requires Acerbi Z1/Z2 + FZ mean + `es_hit_count` key presence whenever nonempty `tail` exposes `es_95`/`realized_es`/`var_95` (NaN ok; empty skips; `tail_es_battery_incomplete:<key>`; scorecard `tail_es_battery_ok`; orthogonal to VaR-battery). Day Wave 42: soft Jackknife+/CV+ `coverage_guarantee_scope` verify requires `marginal_exchangeable` whenever nonempty `jackknife_plus`/`cv_plus` exposes `coverage`/`coverage_floor` (NaN ok; empty skips; `coverage_guarantee_scope_missing:<fam>` / `coverage_guarantee_scope_invalid:<fam>`; no forged scorecard flag). Day Wave 44: soft scorecard forge verify requires that `executed`/`nonempty`/`finite_observation` True flags match catalog helpers (empty `{}` / all-NaN → forge tokens `scorecard_executed_flag_forged:<fam>` / `scorecard_nonempty_flag_forged:<fam>` / `scorecard_finite_observation_flag_forged:<fam>`; False/absent not forgeries).

The same research hook exposes Acerbi--Székely (2014) ES diagnostics
(`acerbi_szekely_z1`, `acerbi_szekely_z2`) when an ES forecast is supplied.
Loss convention: positive losses; hit when \(L_t > \mathrm{VaR}_t\). With
tail probability \(p=1-\alpha_{\mathrm{VaR}}\):

\[
Z_1 = \frac{1}{N p}\sum_t 1_{L_t>\mathrm{VaR}_t}\frac{L_t}{\mathrm{ES}_t} - 1,\qquad
Z_2 = \mathrm{mean}_{t:L_t>\mathrm{VaR}_t}\Bigl(\frac{L_t}{\mathrm{ES}_t}\Bigr) - 1
\]

Under a correct forecast, \(\mathbb{E}[Z_1]=\mathbb{E}[Z_2]=0\). Empty / no hits /
non-positive ES on hits → honest NaN; bad \(p\) → fail-closed. Wired through
`var_backtest_hooks` / book diagnostics **and** the research `bench_tail` family
blob (Day Wave 15: primary + `*_unscaled` / `*_scaled` mirrors beside Kupiec) as
research fields only — never promotion or live-readiness evidence.

## Fissler–Ziegel joint VaR/ES scoring (research-only)

Positive losses \(L=-R\). Let \(\alpha\in(0,1)\) be the VaR **coverage** level
(e.g. 0.95), so \(\mathrm{VaR}_\alpha=q_\alpha(L)\) and breach rate is \(1-\alpha\).
Forecasts \((v,e)=(\mathrm{VaR},\mathrm{ES})\) with \(e>0\).

Nolde–Ziegel / Fissler–Ziegel **FZ0** member (\(G_1\equiv 0\), \(G_2=\log\)):

\[
S(v,e;L)=\frac{1}{1-\alpha}\,1_{L>v}\frac{L-v}{e}+\frac{v}{e}-1+\log(e)
\]

Lower expected score is better (jointly consistent for \((\mathrm{VaR}_\alpha,\mathrm{ES}_\alpha)\)).
**Correction (bugbot, 2026-09-16):** an earlier variant omitted the \(1/(1-\alpha)\)
scaling on the hit term and applied an extra \((1-\alpha)\) to \(v/e\). That score
is *not* proper — its expectation is minimized at \(e=(1-\alpha)\cdot\mathrm{ES}\)
rather than \(\mathrm{ES}\) — and both this spec and the implementation now use the
canonical form above.
Implemented as `fissler_ziegel_loss` / `mean_fissler_ziegel` in `metrics.scoring`
and optionally surfaced as `fissler_ziegel_mean` on `var_backtest_hooks` when an
ES forecast is supplied. Day Wave 15 also surfaces the same key on research
`bench_tail` (coverage \(\alpha\) for FZ; miss level \(1-\alpha\) for Acerbi Z1 —
matching analytics). Empty / non-positive ES / non-finite → honest NaN;
bad \(\alpha\) / length mismatch → fail-closed. Research-diagnostic only — never
live capital or promotion evidence.

## Book diagnostics (execution / risk — not research headlines)

For a simulated or paper equity path with returns \(R_t\):

- Gross \(\sum_i |w_i|\), net \(\sum_i w_i\)
- Turnover \(\sum_i |w_{i,t}-w_{i,t-1}|\) (or executed notional / NAV)
- Capacity proxy: participation \(|\mathrm{trade\ notional}|/\mathrm{ADV}\)
- Realized vol, max drawdown, Calmar as **execution diagnostics only** (`role=1`)
- Historical VaR/ES on \(L=-R\) (Acerbi–Tasche discrete ES)
- Stylized stress: \(\pm\sigma\) shock, liquidity haircut, **liquidity freeze**, **gap-open** (kσ overnight), correlation-spike 1σ P&L

When `data_source=SYNTHETIC`, all of the above are research-only. Never live P&L evidence.

**Dual honesty catalogs:** research family/scorecard blobs use
`FORBIDDEN_RESEARCH_METRIC_KEYS` / `family_blob_forbidden_metrics_absent` (no
sharpe/sortino/calmar/pnl/nav *key tokens* in lab headlines). Paper/backtest
`analytics_export` may nest equity `nav_*` and stress `*_pnl` diagnostics under
the shared `ANALYTICS_SCHEMA_KEYS`, but `live_pnl_claim` must stay false —
`validate_analytics_export` fails closed on `live_pnl_claim=true`. Do not apply
the research forbidden-key scanner to the full export blob.

## Optimizer turnover + ADV capacity (hard + soft)

- Hard turnover: \(\|w - w_{\mathrm{prev}}\|_1 \le\) `constraints.turnover_limit`
- Soft TC: `lambda_tc * tc · |Δw|` (linear cost vector)
- Soft turnover: `lambda_turnover * ||Δw||_1` (explicit L1 penalty; default 0)
- **Hard ADV capacity**: `|Δw_i| * nav / ADV_i ≤ max_adv_participation` when `adv_dollars`
  is passed to `optimize_mean_variance` and `max_adv_participation` is not `None`.
  Default `0.10` (10% of ADV per name per rebalance). Set to `None` to disable.
- **Soft / diagnostic capacity**: `metrics.analytics.capacity_proxy` reports mean/max/p95
  participation `|trade notional|/ADV` without changing weights. Limits: requires finite
  positive ADV; empty → NaNs with `n=0`.

Limits note: hard ADV is per-name on *weight change*, not on absolute weight; tiny ADV
or huge NAV makes the bound extremely tight (can pin `w ≈ w_prev`). Soft proxy never
rejects — it only diagnoses.

## Northset (candles + L2)

OHLC identities (fail-closed data contract):

\[
H_t \ge \max(O_t, C_t, L_t),\quad L_t \le \min(O_t, C_t, H_t),\quad H_t \ge L_t
\]

Parkinson (1980) range variance vs close-to-close realized variance, scored with QLIKE:

\[
\hat\sigma^2_{\mathrm{Park}} = \frac{(\ln H_t - \ln L_t)^2}{4\ln 2},\qquad
\hat\sigma^2_{\mathrm{cc}} = \bigl(\ln(C_t/C_{t-1})\bigr)^2
\]

Garman–Klass (1980):

\[
\hat\sigma^2_{\mathrm{GK}} = \tfrac12 (\ln H_t/L_t)^2 - (2\ln 2 - 1)(\ln C_t/O_t)^2
\]

Rogers–Satchell (1991):

\[
\hat\sigma^2_{\mathrm{RS}} = \ln(H_t/C_t)\ln(H_t/O_t) + \ln(L_t/C_t)\ln(L_t/O_t)
\]

Yang–Zhang (2000) combines overnight, open-to-close, and RS with weight \(k = 0.34/(1.34+(n+1)/(n-1))\).
Receipt `yang_zhang_variance` is that pooled sample statistic. Receipt
`yang_zhang_qlike_vs_cc` is QLIKE of close-to-close RV vs a **per-security
expanding** Yang–Zhang forecast (bars strictly before \(t\)); it is not an
in-sample pooled constant.

Receipt `corwin_schultz_spread` (OHLC estimator ≠ touch means). Corwin–Schultz (2012) two-day high-low spread \(S = 2(e^\alpha-1)/(1+e^\alpha)\) on PIT pairs \((t-1,t)\).

Cont–Kukanov–Stoikov OFI on consecutive top-of-book snapshots (daily sampling on SYNTHETIC L2).

Amihud illiquidity \(|R_t|/(C_t V_t)\).

Kyle (1985) λ from OLS \(\Delta m_t = \lambda q_t + \varepsilon_t\) where \(q_t\) is signed depth (\(D^{\mathrm{bid}} - D^{\mathrm{ask}}\)).

Northset receipt: `roll_spread` is this panel estimator (mid-based), **not** interchangeable with touch `mean_quoted_spread` / `mean_spread_bps` (see DATA_CONTRACTS **Touch means vs OHLC**). Roll (1984) implied spread \(2\sqrt{-\gamma_1}\) from lag-1 autocovariance of mid changes; undefined (NaN) when \(\gamma_1 \ge 0\).

Imbalance / microprice / wick / OFI / CLV / VPIN / fused candle×LOB feature ICs are **date-level**:

\[
\mathrm{RankIC}_t = \mathrm{Spearman}_i(\hat s_{i,t},\, y_{i,t+1}),\qquad
y_{i,t+1} = C_{i,t+1}/C_{i,t} - 1
\]

Mean of \(\{\mathrm{RankIC}_t\}\) with Newey–West HAC \(t\)/\(p\) (`date_ic_series`). Dates with fewer than `min_names` finite pairs are dropped; fewer than 3 kept dates → NaN inference. Names on the same date are never stacked as i.i.d. rows. Candle+LOB bench sets `ic_method=date_level_spearman_hac` and `research_only=true` (no Sharpe/pnl keys).

Receipt `abdi_ranaldo_spread` (OHLC estimator ≠ touch means). Abdi–Ranaldo (2017) close-high-low spread \(S=2\sqrt{\max(\eta,0)}\) with \(\eta=(C_t-m_t)(C_{t-1}-m_{t-1})\) and \(m=(H+L)/2\).

Overnight share \(\overline{(\ln O_t/C_{t-1})^2}/\overline{\hat\sigma^2_{\mathrm{cc}}}\). Split forecast \(\hat\sigma^2_{\mathrm{on}}+\hat\sigma^2_{\mathrm{oc}}\) vs Parkinson is Diebold–Mariano on date-level QLIKE.

Session BNS jump share \(\max(0,1-\mathrm{BV}/\mathrm{RV})\) with \(\mathrm{BV}=(\pi/2)\sum |r_i||r_{i-1}|\).

Session chain identity: session close at index \(i\) equals session open at \(i+1\).

Liquidity sweep (Osler 2003 stop-cluster motivation): with prior \(L\)-bar extremes
\(\bar H_t = \max_{i<t} H_i\), \(\underline L_t = \min_{i<t} L_i\) (shifted, PIT), a high
sweep is \(H_t > \bar H_t\); it reclaims when \(C_t < \bar H_t\) and follows through when
\(C_t \ge \bar H_t\) (symmetrically for lows). Sweep depth \((H_t-\bar H_t)/\bar H_t\);
signed reject/follow depth enters date-level IC.

The institutional sweep event study uses no event-close fill. An event known at
close \(t\) enters at \(O_{t+1}\), exits at \(C_{t+h}\), and removes the same-date
cross-sectional mean:

\[
r^{xs}_{i,t,h} = \frac{C_{i,t+h}}{O_{i,t+1}} - 1
- \frac{1}{N_t}\sum_j\left(\frac{C_{j,t+h}}{O_{j,t+1}} - 1\right).
\]

Reclaim direction is positive for low reclaims and negative for high reclaims;
follow-through direction is positive for high breaks and negative for low breaks.
Event observations are equal-weighted within date. HAC, circular-block bootstrap,
and chronological folds then run on the **panel calendar** with idle dates filled
at 0 (`inference_index=calendar_including_idle_zeros`); compressing to event dates
would treat sparse events as consecutive days and overstate precision.
`mean_excess_bps` remains the event-day mean. Invalid OHLC prints are quarantined
before the PIT rolling extreme so they cannot mint flags or pollute prior high/low.
Next-open entry and horizon-exit prices are also dropped when those bars fail
OHLC identities.
Yang–Zhang QLIKE vs close-to-close is per-security expanding (PIT), not an in-sample
pooled constant; VPIN remains a count-window bulk-OFI proxy until true prints exist.

Matched controls (H44/H45): with direction \(s \in \{+1,-1\}\), per date

\[
\Delta_t = \operatorname{mean}_s\, s\left(\bar r^{\,xs}_{\mathrm{event},s,t} - \bar r^{\,xs}_{\mathrm{control},t}\right),
\]

where controls are same-date sweep-eligible names that swept neither side; HAC t
on the calendar-embedded \(\{\Delta_t\}\) (idle dates 0); `mean_diff_bps` stays
event-conditional. The lead diagnostic tests the signal against the *previous* bar's
cross-sectional excess return (pre-trend disclosure / lookahead canary). The parameter
grid re-runs the horizon-1 study across detector lookbacks and reports every cell as a
counted trial with a sign-stability flag. Liquidity-quartile matching repeats the
control difference inside lagged dollar-volume bins (H46/H47). Name-clustered
inference collapses each security to one mean and uses an iid t (`lags=0`; H49).
Two-way clustered inference (H50) uses every event observation and Cameron–Gelbach–Miller
``V = V_{\mathrm{date}} + V_{\mathrm{name}} - V_{\mathrm{white}}``
(`inference_index=event_rows_not_calendar_zeros`). A restricted Rademacher
wild-cluster bootstrap on dates is the few-cluster honesty companion
(`sweep_*_two_way_wild_p`), not a second discovery claim.
The untradeable overnight gap (H51) is signed excess of
``O_{t+1}/C_t - 1`` on the calendar including idle zeros — the jump next-open
entry cannot capture. Corwin–Schultz pairs are ``(t-1, t)`` (known at close t).
The last chronological fold is a frozen out-of-time holdout (H48 same-sign bound).
H45 is the predeclared primary executable test; event-study BH-FDR is a secondary
family. Event ADV participation is `participation_rate \times` event dollar volume
over lagged ADV (crowding disclosure, not a live capacity claim). H33/H34 remain
descriptive date-level ICs of signed depth, not executable studies.

Implementation map: flags/signed depths — `northset.sweeps.liquidity_sweep_frame` / `sweep_rates`; forward XS returns — `sweep_research.sweep_forward_frame` (`sweep_excess_ret_{h}`); battery — `sweep_evidence_battery` with `event_studies`, `volatility_regimes`, `permutation_placebos` (`within_date_permutation_test`), BH-FDR, and `estimated_round_trip_cost_bps`. Signals under test: `sweep_reject_signed`, `sweep_follow_signed` only.


SYNTHETIC L2 (`synthesize_l2_from_bars` / `synthesize_session_l2`) is labeled plumbing (`source=synthetic`, `revision_id=SYNTHETIC_LOB_v1`). Session L2 is multi-snapshot **synthetic** path stats, not exchange clock L2. Vendor Alpaca/Polygon remap is offline column mapping onto the same panel schema — still `research_only`; not a live tape or execution claim.

Kyle / OFI research family (`northset.kyle_ofi`): Cont–Kukanov–Stoikov OFI on consecutive
tops; per-date Kyle \(\lambda\) from OLS \(\Delta m = \lambda q + \varepsilon\) with
\(q\in\{\mathrm{signed\_depth},\mathrm{OFI}\}\); HAC on the \(\lambda\) series; date-level
Spearman/Pearson of flow→\(\Delta m\) and lag-0/lag-1 OFI→\(\Delta m\) via `date_ic_series`.
Family blob `kyle_ofi` is `research_only` (no Sharpe/pnl/nav keys). Optional: `northset.include_kyle_ofi` nests that blob under the Northset receipt as `kyle_ofi` (default off); not a separate catalog-required family.

**Flow alias (same formula, two labels):** imbalance \(q_{\mathrm{imb}} = \mathrm{bid\_depth}-\mathrm{ask\_depth}\) is column `signed_depth` on the kyle_ofi fuse and column `signed_volume` on the always-on `bench_northset` fuse into `_panel_kyle`. Always-on receipt scalars `kyle_lambda` / `kyle_r2` use per-name OLS then cross-name mean on `signed_volume`; nested nest uses date-level \(\lambda\) + HAC on `signed_depth`. Do not equate the two estimator paths or treat the rename as a new signal. See DATA_CONTRACTS fuse-alias table and NORTHSET disambiguation.

Integrity (not alpha): `validate_session_book_counts` enforces constant session-snap cardinality;
`join_coverage` fail-closes sparse candle↔book asof joins; `validate_book_panel_depth_honesty`
forbids finite slopes on thin books and NaN slopes on deep books.

Book snapshot aliases: `half_spread=spread/2`, `quoted_spread_bps≡spread_bps`, `touch_size_imbalance≡imbalance_top`; `microprice_weight_balance` is the microprice convex weight on the ask. Northset fuse **FIXED dual columns:** book `effective_spread` (ask−bid) preserved; candle diagnostic `close_mid_abs_rel` = `2|C−mid|/mid`. Receipts: `mean_effective_spread` = book alias; `mean_close_mid_abs_rel` = candle; `mean_spread_bps` / `mean_quoted_spread` = quoted. See DATA_CONTRACTS **dual columns**. `depth_imbalance_abs=|imbalance_depth|`; `spread_over_mid=spread/mid`.

Book depth shape (`book_metrics.DEPTH_SHAPE_FIELDS`): log-size slopes, log-price slopes (Commander residual #2), and mean log tick spacings (residual #3). Finite only when that side has \(n\ge 2\) positive finite levels; otherwise NaN. Zero adjacent price gaps → NaN tick spacing even on deep books. `validate_book_panel_depth_honesty` requires thin→NaN for all present shape fields and deep→finite for size/price slopes.


## Paper ledger schema (v2)

Persisted under `data/metadata/paper/<run_id>/`: `orders.parquet`, `equity.parquet`, `positions.parquet`, `cash_ledger.parquet`, `broker_state.json` (resume), `promotion_dry_run.json` (shadow→champion dry-run; `would_promote_live` always false), `meta.json`.

### Queue imbalance vs OFI (Northset)

`queue_imbalance` = TOB size imbalance (level); `ofi` = Cont-style top-size change (flow). Receipt `queue_imbalance_mean` is not mean OFI. DATA_CONTRACTS **queue_imbalance vs ofi**.

### Session OFI sum vs daily OFI

`session_ofi_sum` aggregates Cont-style OFI across session L2 snapshots within a parent day; always-on `ofi`/`ofi_lag` are the daily Cont (or attach-proxy) path. Receipt `session_ofi_sum_mean` ≠ mean daily ofi. See DATA_CONTRACTS.

### Amihud / volume_over_range / true_range

Northset joins bar Amihud, volume/range, and Wilder true range; receipts `amihud_mean` and `mean_true_range` (no plain `volume_over_range_mean`). Not touch or Roll/CS/AR spreads.

### Illiquidity means (honesty)

Northset `amihud_mean` / `mean_true_range` / volume_over_range ICs are `research_only` diagnostics — **no live Sharpe**. Session OFI vs bar OFI/queue: DATA_CONTRACTS **Session OFI vs bar-level**.

### VPIN family (Northset)

Daily `vpin_proxy` → `vpin_mean` / H32; session-book `|ofi_sum|/|ofi|_abs_sum` → `session_book_vpin_mean` / H43; `session_bulk_vpin` is candle-volume VPIN without an H-id. DATA_CONTRACTS **session_book_vpin_mean vs vpin_mean**.

## robinhood+ (Kronos K-line engine)

Two-stage K-line language model (Shi et al., 2025). Split-adjusted
OHLCV+amount \(x_t\in\mathbb{R}^6\) is z-scored on the causal lookback,
projected by a seeded orthonormal map \(W\), L2-normalized, and quantized
to bipolar bits. Coarse tokens \(s1\) use the first \(b_1\) bits; fine
tokens \(s2\) use the remaining \(b_2\) bits (LSB-first, matching Kronos
BSQ). The default decoder samples \(s1_{t+1}\mid(s1_t,s2_t)\) then
\(s2_{t+1}\mid(s1_t,s2_t,s1_{t+1})\) from Laplace-smoothed lookback
counts (temperature + nucleus). Decode inverts bits \(\to\) latent
\(\to\) \(W^\top\) \(\to\) denormalize and repairs the OHLC identity
\(\mathrm{high}\ge\max(\mathrm{open},\mathrm{close})\),
\(\mathrm{low}\le\min(\mathrm{open},\mathrm{close})\).

Horizon simple returns use the last lookback close \(P_t\) and the
sampled close at step \(h\):

\[
\hat R_{t,t+h}^{(m)} = \frac{\hat P_{t+h}^{(m)}}{P_t}-1
\]

Expected return, quantiles, and \(\mathbb{P}(\hat R>0)\) are sample
moments over \(m=1,\ldots,M\) paths. Only horizons \(h\le\) `pred_len`
are emitted. Bars with `event_time` (and `available_time` when present)
after the decision clock are excluded. This is a research-lab path
forecast, not a live P&L claim.

## Kelly–Malamud–Zhou random Fourier ridge (`rff`)

JoF 2024 equation (20) (NBER w30217 eq. 21) maps standardized public
cross-sectional features \(G\) to paired random Fourier features

\[
S_i = \bigl[\sin(\gamma\omega_i'G),\;\cos(\gamma\omega_i'G)\bigr]',
\quad \omega_i\sim\mathrm{i.i.d.}\,N(0,I),\;\gamma=2.
\]

\(P\) is even (`train.rff_n_features`). Columns of \(S\) are standardized
on the training sample (NBER footnote 36); the JoF display omits
\(P^{-1/2}\) because that scale is absorbed by column standardization
and \(z\). Ridge uses the paper's parameterization, not sklearn
`alpha`:

\[
\hat\beta(z)=\bigl(zI+T^{-1}S'S\bigr)^{-1}T^{-1}S'R.
\]

When \(P>T\), the dual \(\beta=S'(SS'+zTI)^{-1}y\) is used. \(z=0\) is
ridgeless (minimum-norm interpolator). **Deviation:** the paper is
time-series market timing on 15 macro predictors; Dipcatcher stacks the
cross-sectional panel and treats \(T\) as the number of finite
stock-date rows. Scores are \(\hat y=\bar R+S_{\mathrm{oos}}'\hat\beta\).
No Sharpe in metadata. Walk-forward is the existing purged date folds.

## Kozak–Nagel–Santosh SDF ridge (`sdf_ridge`)

Managed portfolios \(F_t=n_t^{-1}Z_t'r_t\) from lagged public CS
features and the ranking label. KNS (22):

\[
\hat b=(\Sigma+zI)^{-1}\mu,\qquad \mu=\bar F,\;\Sigma=\widehat{\mathrm{Cov}}(F).
\]

In PC space the shrinkage factor on OLS is \(d_j/(d_j+z)\), which is
stronger for small eigenvalues — that *is* the extra shrinkage, not a
second hyperparameter. Stock scores are \(Z\hat b\). Dates are required.
**Deviation:** KNS estimate a monthly SDF on characteristic-managed
factors; here \(r_t\) is the ranking target (e.g. 5-day excess), not a
tradable monthly excess return.

## Kelly–Pruitt–Su IPCA (`ipca`)

Restricted IPCA (\(\Gamma_\alpha=0\)):

\[
r_{t+1}=Z_t\Gamma f_{t+1}+\varepsilon_{t+1},\qquad \Gamma'\Gamma=I_K.
\]

ALS alternates FOC (6) \(f_{t+1}=(\Gamma'Z_t'Z_t\Gamma)^{-1}\Gamma'Z_t'r_{t+1}\)
and FOC (7) \(\mathrm{vec}(\Gamma)=( \sum_t f_t f_t'\otimes Z_t'Z_t)^{-1}\sum_t (f_t\otimes Z_t')r_{t+1}\)
until \(\max|\Delta\Gamma|<10^{-6}\) (default). Identification: thin QR,
diagonal descending \(\mathrm{Cov}(f)\), non-negative mean \(f\).
Initialization: leading eigenvectors of \(\sum_t x_t x_t'\) with
\(x_t=Z_t'r_{t+1}\). Restricted predictor is \(Z\Gamma\mu_f\). Catalog
Unrestricted IPCA (`ipca_alpha`) jointly estimates \((\Gamma_\alpha,\Gamma)\)
with augmented factors \(F_{\mathrm{aug},t}=(1,f_t)'\):

\[
r_{t+1}=Z_t\Gamma_\alpha+Z_t\Gamma f_{t+1}+\varepsilon_{t+1},\qquad
\mathrm{vec}(\Gamma_{\mathrm{aug}})=\Bigl(\sum_t F_{\mathrm{aug},t}F_{\mathrm{aug},t}'\otimes Z_t'Z_t\Bigr)^{-1}\sum_t\bigl(F_{\mathrm{aug},t}\otimes Z_t'\bigr)r_{t+1}.
\]

ALS warm-starts from the restricted solution, then alternates
\(f_t=(\Gamma'Z_t'Z_t\Gamma)^{-1}\Gamma'Z_t'(r_t-Z_t\Gamma_\alpha)\) with the
packed \(\Gamma_{\mathrm{aug}}=[\Gamma_\alpha\mid\Gamma]\) FOC until
\(\max|\Delta\Gamma|,|\Delta\Gamma_\alpha|<10^{-6}\). Identification
(QR, descending \(\mathrm{Cov}(f)\), non-negative mean \(f\)) is applied
to \(\Gamma\) only. Predictor \(Z(\Gamma_\alpha+\Gamma\mu_f)\). Default
\(K=3\). Dates are required. **Deviation:** instruments are the public
CS-z columns, not the paper's 36 firm characteristics split into level
and deviation. No bootstrap pricing test. Research diagnostic, not a
live SDF claim.

## Kozak–Nagel–Santosh SDF elastic net (`sdf_en`)

KNS (28) minimizes the HJ-distance plus \(\ell_2\) and \(\ell_1\):

\[
\hat b=\arg\min_b\,(\mu-\Sigma b)'\Sigma^{-1}(\mu-\Sigma b)+\gamma_2\|b\|_2^2+\gamma_1\|b\|_1.
\]

Implemented as ISTA on the equivalent smooth gradient \(2\Sigma b-2\mu+2\gamma_2 b\)
with soft-thresholding. **Deviation:** fixed \(\gamma_1,\gamma_2\)
(`sdf_en_l1`, `sdf_en_l2`), not LARS-EN with Sharpe-prior \(\kappa\).
If ISTA returns the zero vector (typical when \(\gamma_1\) dwarfs
\(\|\mu\|\)), the ranker retries at \(\gamma_1=0\) rather than emit a
constant score.

## Lettau–Pelger RP-PCA (`rp_pca`)

On the \(T\times L\) managed-portfolio matrix \(X\),

\[
S_{\mathrm{RP}}=\tfrac1T X'X+\gamma\bar X\bar X',\qquad \gamma=-1\text{ is covariance PCA}.
\]

Default \(\gamma=10\) (over-weight means). Loadings \(\Lambda\) are the
leading \(K\) eigenvectors; scores \(Z\Lambda\mu_f\). **Deviation:**
applied to characteristic-managed portfolios of public CS features, not
the paper's characteristic-sorted test assets. No Sharpe of the factors
is stored.

## Giglio–Xiu three-pass (`gx3pass`)

Pass 1: PCA of managed-portfolio returns. Pass 2: \(\lambda_{\mathrm{PCA}}=V_K'\mu\).
Pass 3: each managed column on the PCs yields \(\eta_j\); characteristic
premium \(\eta_j'\lambda_{\mathrm{PCA}}\). Scores \(Z\hat\gamma\).
**Deviation:** test assets are the \(L\) managed portfolios, not a large
equity-portfolio panel. Weak-factor caveats of PCA remain.

## Freyberger–Neuhierl–Weber adaptive group LASSO (`fnw`)

Date-level rank transform of each characteristic to \((0,1)\). Quadratic
spline basis (FNW 4): \(1,c,c^2,\max(c-t_l,0)^2\) with equally spaced
knots. Two-step adaptive group LASSO (5)–(7) then OLS on selected spline
groups. **Deviation:** one global intercept (not \(p_1=1\) inside every
group); \(\lambda\) is configured (`fnw_lam`), not Yuan–Lin BIC. If the
adaptive step selects no characteristic, OLS is run on every spline group
rather than scoring a constant intercept.

## Feng–Giglio–Xiu / BCH double selection (`ds_lasso`)

Columns are standardized (same as PCR / alasso). LASSO of \(y\) on \(Z\),
then LASSO of each selected column on the rest; OLS on the union.
sklearn coordinate descent uses the Gram matrix (\(p\times p\)), not the
naive \(n\)-path. **Deviation:** stock-level ranking label, not a
Fama–MacBeth test of a new traded factor. Post-selection OLS is the
prediction map. Unstandardized pooled OLS on mixed-scale CS columns
produced \(|\hat\beta|\sim10^{-13}\) (a constant score) on the 5-day tape.

## Fama–MacBeth (`fm`)

For each date \(t\) with enough names,

\[
r_{i,t}=a_t+Z_{i,t}\lambda_t+e_{i,t},\qquad
\hat\lambda=\frac1T\sum_t\hat\lambda_t.
\]

Scores are \(Z\hat\lambda\) (the intercept does not rank). **Deviation:**
one pooled window per walk-forward fold, not overlapping monthly FM with
Newey–West on \(\lambda_t\). Public CS-z columns, not the original FM
market-beta specification.

## Gu–Kelly–Xiu PCR (`pcr`) and PLS (`pls`)

NBER w25398 / RFS 2020. Column-standardize \(Z\). PCR takes the leading
\(K\) right singular vectors \(\Omega_K\) of \(Z\) and OLS of \(y\) on
\(Z\Omega_K\). PLS is sklearn SIMPLS (de Jong 1993), the GKX
implementation; Kelly–Pruitt (2015) show PLS is 3PRF without second-pass
intercepts. Default \(K=3\). **Deviation:** \(K\) is configured
(`pcr_n_factors`, `pls_n_factors`), not validation-tuned; Huber loss is
not used on these two linear reducers. Neural nets stay blocked (ADR-007).

## Kelly–Pruitt three-pass regression filter (`tprf`)

JoE 2015 Table 1 with Table 2 automatic proxies. Predictors are
column-standardized. Proxy 1 is the target \(y\); proxy \(k\) is the
residual of the \((k-1)\)-proxy 3PRF. Pass 1: each characteristic on the
proxies (with intercept). Pass 2: each row's characteristic vector on
\(\hat\Phi\) (with intercept). Pass 3: \(y\) on \(\hat F\). OOS uses
frozen \(\hat\Phi\) and \(\hat\beta\). **Deviation:** the paper's \(T\times N\)
is calendar time by many predictors; here rows are stacked stock-dates and
\(N\) is the public CS width (same stacked-\(T\) adaptation as VoC).

## Gu–Kelly–Xiu GBRT (`gbrt`)

Shallow Huber gradient-boosted trees (GKX Algorithm 4 / GBRT+H):
`max_depth=2`, shrinkage \(\nu=\) `gbrt_learning_rate`, \(B=\)
`gbrt_n_estimators`, subsample 0.8. **Deviation:** hyperparameters are
configured, not validation-path tuned; this is not the paper's 94-characteristic
monthly CRSP panel. Random forests are omitted (same tree class). Neural
nets are not implemented (ADR-007). Linear GKX autoencoder remains IPCA.

## Kelly–Malamud–Pedersen principal portfolios (`pp`)

JoF 2023 / NBER w27388. Own-signal \(S=Z\hat\beta_{\mathrm{OLS}}\).
Unbalanced-panel estimator

\[
\hat\Pi=\mathrm{average}_t\, r_t S_t'
\]

over names present on that date (\(r_t\) is the already-aligned ranking
label). Rank-\(K\) SVD \(\hat\Pi_K=U_K\Lambda_K V_K'\). Date-\(t\) scores
are \(\hat\Pi_K S_t\) in name order; names unseen in training keep the
own-signal. **Deviation:** \(S\) is the pooled OLS fitted value, not a
single characteristic such as momentum; PEPs/PAPs (symmetric vs
antisymmetric split) are not stored as separate book weights. This does
not size the book.

## Rapach–Strauss–Zhou combination (`combo`)

Equal-weight average of \(L\) univariate OLS forecasts
\(\hat r^{(j)}=a_j+Z_{\cdot j}b_j\). **Deviation:** Rapach et al. combine
equity-premium time-series models; here each “model” is a public CS
characteristic. Intercepts do not change cross-sectional rank.

## Zou adaptive LASSO (`alasso`)

Columns are standardized. First-stage OLS weights
\(w_j=|\hat\beta_j^{\mathrm{OLS}}|^\gamma/\max_k|\hat\beta_k^{\mathrm{OLS}}|^\gamma\)
(\(\gamma=1\)), floored at \(10^{-3}\) so a near-zero slope cannot divide
its column by \(10^{-8}\) and stall coordinate descent. LASSO on
\(Z_{\cdot j}/w_j\), then \(\hat\beta_j=\hat\theta_j/w_j\). The L1 step
uses Gram-precomputed coordinate descent (same \(p\times p\) path as
`ds_lasso`); the naive \(n\)-path could not finish an expanding 5-day
horse race. **Deviation:**
\(\lambda=\)`alasso_alpha`\(\cdot\sigma_y\) (target-sd units, see below), not
BIC/CV; one pooled window per fold, not Zou’s oracle-rate asymptotics as
a live claim.

## \(\ell_1\) penalties in target-sd units (Wave 147)

`ds_lasso_alpha`, `alasso_alpha`, and `fnw_lam` are quoted in units of the
regressand’s standard deviation: the absolute penalty passed to the
solver is \(\alpha\,\hat\sigma_y\) (for the FGX treatment LASSOs of
\(Z_{\cdot j}\) on \(Z_{\cdot -j}\), \(\alpha\,\hat\sigma_{Z_j}\)). A fixed
absolute \(\alpha=0.01\) is mild on a unit-variance test target and
zeroes every coefficient on a 5-day idiosyncratic return with
\(\sigma_y\approx0.03\), which silently turned `fnw` and `ds_lasso` into
their OLS fallbacks on the file tape. The knobs did not change; their
units did.

## Public bar-characteristic zoo (Wave 147)

`PUBLIC_FEATURES` is 43 columns, all functions of OHLCV, sector, and
membership at or before \(t\): CS-z of returns / momentum (5, 20, 60,
126, 12-1, 20-skip-5, residual 20), reversal and \(z\) vs MA20,
overnight vs intraday returns (1 and 20 sessions), MAX / MIN 20,
realized skew / kurtosis 20, volatility (20, 60, EWMA, Parkinson,
Garman–Klass, vol-of-vol, downside, 20/60 ratio), trailing 60-session
OLS beta and idiosyncratic vol vs the benchmark, liquidity
(Amihud 20 / 60, ADV, dollar volume, relative volume, turnover proxy,
volume vol, ADV 20/60 ratio, log price), plus rank-space `cs_pct`
reversal and overnight. Long-lookback columns
(`mom_126`, `mom_12_1`, `high_52w_prox`) take the cross-sectional
neutral value 0 when unavailable (GKX median-fill); every other null
drops the row.

Robust cross-sectional \(z\) is \((x-\mathrm{med})/(1.4826\,\mathrm{MAD})\).
When more than half the cross-section shares one value MAD is exactly 0;
the previous \(10^{-12}\) floor produced \(|z|\sim10^{11}\)
(`cs_z_high_52w_prox`, `cs_z_ret_overnight` on the 55-name tape) and
destroyed every OLS-based ranker. The scale now falls back to the group
standard deviation, then to “no dispersion → \(z=0\)”. Feature set
version `features.v4`.

Walk-forward purging for an \(h\)-bar label uses `horizon_bars=h`, and
date-IC HAC lags are `overlap_aware_hac_lags(n_dates, h)`, so a 5-day
label is not scored as if it were 1-day.

## Pooled ridge in date units (Wave 149)

sklearn `Ridge` minimises un-normalised RSS \(+\alpha\|b\|^2\). On a
stacked tape \(n\sim10^5\), \(\alpha=1\) is OLS. When dates are passed,
\(y\) is date-demeaned and \(\alpha_{\mathrm{used}}=\alpha T\) so the
pooled Gram is \(X'X+T\alpha I\), the sum of date-level `Ridge(α)`
problems. Unit tests that omit dates keep \(\alpha\) unchanged.

## Ridge Fama–MacBeth (`fm_ridge`)

Per-date CS ridge (intercept dropped from the score), then
\(\hat\lambda=\mathrm{mean}_t\hat\lambda_t\). OLS FM skips a date unless
\(N_t\ge p+2\); ridge runs at \(N_t\ge 8\). This is the identified
small-\(N\) cousin of `fm`.

## Classic signed characteristics (`classic`)

Fixed signs, no estimated slopes: \(+\) reversal, skip-momentum,
residual momentum, 12-1, 52-week-high proximity, Amihud; \(-\) MAX,
idio vol. If feature names are omitted, equal weight (unit tests). If
names are passed and none match, the ranker raises. Not OOS-tuned.

## Short-horizon daily CS (`reversal`, `classic_st`, `ridge_st`, `fm_st`, `combo_ic_st`)

A priori daily/weekly subset, not the monthly zoo. `reversal` is
Jegadeesh \(+\mathrm{cs\_z\_reversal\_1}\) only. `classic_st` signs:
\(+\mathrm{cs\_z\_reversal\_1}\), \(-\mathrm{cs\_z\_ret\_5}\) (Lehmann
weekly reversal), \(+\mathrm{cs\_z\_mom\_skip\_5\_20}\),
\(+\mathrm{cs\_z\_idio\_mom\_20}\), \(-\mathrm{cs\_z\_max\_ret\_20}\),
\(-\mathrm{cs\_z\_idio\_vol\_60}\). `ridge_st` / `fm_st` / `combo_ic_st`
estimate slopes on `SHORT_HORIZON_FEATURES` (12 daily/weekly columns).
Signs and the column mask are frozen before OOS.

## IC-weighted combination (`combo_ic`)

Rapach univariate OLS forecasts weighted by \(\max(\overline{\mathrm{IC}}_j,0)\)
computed on the **train** dates of the fold. If every train IC is
negative, equal weight. Not a holdout IC weight.

## Discounted MSFE combination (`combo_msfe`)

Rapach–Strauss–Zhou discounted MSFE weights on univariate CS OLS.
Within each train fold, the last 25% of dates are a nested holdout
(in-sample MSE if the fold is too short). Univariate OLS is fit on the
inner train; date-level MSE on the nested holdout is discounted with
\(\theta=0.99\). Combination weights are \(w_j \propto 1/\mathrm{MSFE}_j\).
Slopes used at predict-time are refit on the full train fold. Not OOS-tuned.

## Rolling daily-CS walk-forward (hedge_lab)

`configs/hedge_lab.yaml` uses `validation.scheme: rolling` with
`train_bars: 252`. Daily reversal is short-memory; expanding 10-year
pooled fits on 54 names mixed decaying premia into champion ridge.
Wide tape inherits the same window. Champion remains public ridge until
pairwise DM of \(-\mathrm{IC}\) plus White RC / SPA / StepM promote a
challenger. `blend_weight` 0.

## Sign-flip mirror books

Let \(r\) be the frictionless dollar-neutral long-short of a score. The
mirror is \(-r\). Sharpe is odd:
\(\mathrm{SR}(-r)=-\mathrm{SR}(r)\). It is also leverage-invariant
(\(c\neq 0\), rf = 0): \(\mathrm{SR}(c r)=\mathrm{sign}(c)\,\mathrm{SR}(r)\).
Modeled spread / commission / impact \(c_t\ge 0\) are even, so both
books realize \(r-c\) and \(-r-c\). Their Sharpes no longer sum to
zero; the sum is typically negative. A 5% drawdown halt
(`dd_limit=0.05`) cannot coexist with \(-150\%\) total return on the
same path. Nested anti-univariate (lowest train date IC) is the same
search as best-train IC after the sign flip. `hedge-lab --mirror`
negates `target_weight`. **Wave 152:** discarded as a book; diagnostic
only. Not a live P&L claim. `blend_weight` 0.

## Size / vol residual ridge (`ridge_neut`)

Within each decision date, OLS-residualize every non-control public
column on `cs_z_adv`, `cs_z_log_price`, and `cs_z_vol_20`, then fit
date-demeaned T-ridge. The transform uses only that date's
cross-section. A priori neutralization, not OOS-tuned. Challenger only;
champion remains public ridge.

## quant-models engines

`quant_fund.quant_models` ports
[davidalmeida90/quant-models](https://github.com/davidalmeida90/quant-models)
and the README sibling repos as research engines (ADR-033).

Black–Scholes–Merton with yield \(q\):

\[
d_1=\frac{\ln(S/K)+(r-q+\sigma^2/2)\tau}{\sigma\sqrt{\tau}},\quad
d_2=d_1-\sigma\sqrt{\tau}
\]

\[
C=Se^{-q\tau}N(d_1)-Ke^{-r\tau}N(d_2)
\]

Implied vol is Brent on that price inside the no-arbitrage bounds.
Greeks are the raw BSM derivatives; desk scaling is `greeks.SCALE`
(vega per vol point, theta per calendar day). Time derivatives are
\(\partial/\partial t=-\partial/\partial\tau\). CRR uses
\(u=e^{\sigma\sqrt{\Delta t}}\), \(d=1/u\),
\(p=(e^{(r-q)\Delta t}-d)/(u-d)\); American nodes take
\(\max(\text{continuation},\text{intrinsic})\). Heston is the
Albrecher little-trap CF, \(P_1,P_2\) by trapezoid. SVI is Gatheral
raw \(w(k)=a+b(\rho(k-m)+\sqrt{(k-m)^2+\sigma^2})\); butterfly uses
the Gatheral–Jacquier \(g(k)\ge 0\). NSS zeros:

\[
y(\tau)=\beta_0+\beta_1\frac{1-e^{-\tau/\lambda_1}}{\tau/\lambda_1}
+\beta_2\left(\frac{1-e^{-\tau/\lambda_1}}{\tau/\lambda_1}
-e^{-\tau/\lambda_1}\right)
+\beta_3\left(\frac{1-e^{-\tau/\lambda_2}}{\tau/\lambda_2}
-e^{-\tau/\lambda_2}\right)
\]

HRP is Lopez de Prado (2016): single-linkage on
\(\sqrt{(1-\rho)/2}\), quasi-diagonalize, recursive bisection with
inverse-variance cluster variance. GEX per contract, dealer sign
(calls \(+\), puts \(-\), an assumption):

\[
\mathrm{GEX}=\mathrm{sign}\cdot\gamma\cdot\mathrm{OI}\cdot 100\cdot S^2\cdot 0.01
\]

Last-hour rule: previous-close net GEX \(<0\) → go *with* the
open-to-15:30 return; GEX \(>0\) → fade (optional). No overnight, no
broker. TSMOM: \(\mathrm{sign}(\sum_{t-L}^{t-s} r)\cdot
\sigma_{\mathrm{target}}/(\sigma\sqrt{252})\). GKX OOS \(R^2\):
\(1-\sum(y-\hat y)^2/\sum y^2\). Krauss window: trailing-date logistic
on public features predicting above-median next-day idio. Discrete
delta-hedge error is gamma × rebalance gap when
\(\sigma_{\mathrm{realised}}=\sigma_{\mathrm{implied}}\). Not a live
P&L claim. `blend_weight` 0. Neural vol / deep hedging remain
ADR-007.

## Lightspeed engines

`quant_fund.lightspeed` ports
[cosmic-hydra/lightspeed](https://github.com/cosmic-hydra/lightspeed)
as research engines (ADR-034). No Alpaca.

SMA-seeded EMA on signal close \(C_t\), seed at bar \(n-1\):

\[
\mathrm{EMA}_n=\mathrm{SMA}_n,\quad
\mathrm{EMA}_t=\alpha C_t+(1-\alpha)\mathrm{EMA}_{t-1},\quad
\alpha=\frac{2}{n+1}
\]

Frozen `tqqq-long-full-v1`: \(n_{\mathrm{fast}}=20\),
\(n_{\mathrm{slow}}=180\). Gap
\(( \mathrm{EMA}^{\mathrm{fast}}_t-\mathrm{EMA}^{\mathrm{slow}}_t)/C_t\).
If `flatten_when_fast_below_slow` and the gap is negative, TQQQ
weight is 0 and the residual is SGOV. Otherwise the raw risk
sleeve is \(\mathrm{clip}(\mathrm{vol\_budget}/\hat\sigma^{\mathrm{QQQ}}_t,0,0.98)\)
with `vol_budget` 10 (saturates at `max_tqqq` in a confirmed
LONG). Rebalance every 5 sessions; `signal_delay_sessions` 1
shifts the executable weight: \(w_t\leftarrow w_{t-1}\) for the
first delay bar, then \(w_t\leftarrow w^{\mathrm{raw}}_{t-d}\).

Frozen nautica / stock-momentum: score
\(C_t/C_{t-63}-1\) (`mom_blend` 1), eligible only if score \(>0\)
and \(C_t>\mathrm{SMA}_{200}\). Crash: a held name with
\(C_t/C_{t-10}-1\le-0.2\) is flattened. Top-1, vol size
\(\mathrm{clip}(0.6/\hat\sigma_i,0,0.95)\), residual SGOV,
delay 1, 10 bp.

AFML metalabel (López de Prado 2018): expanding-window logistic
\(P(\text{side correct}\mid\text{signal})\). Multiplier
\(\mathbf{1}\{P\ge\tau\}\max(2P-1,0)\in[0,1]\). Reduce-only.

CS challenger `nautica`: a priori \(+1\) on `cs_z_mom_60`. No
estimated slopes. Champion remains public ridge. `blend_weight` 0.
Not a live P&L claim.

## Discrete HMM (Jurafsky & Martin SLP3 Appendix A)

`quant_fund.hmm` implements the discrete first-order HMM from
https://web.stanford.edu/~jurafsky/slp3/A.pdf (Eisner ice-cream
running example). This is not `hmmlearn`'s Gaussian regime model.

\[
\alpha_1(j)=\pi_j b_j(o_1),\quad
\alpha_t(j)=\sum_i \alpha_{t-1}(i)a_{ij}b_j(o_t),\quad
P(O\mid\lambda)=\sum_i \alpha_T(i)
\]

\[
v_t(j)=\max_i v_{t-1}(i)a_{ij}b_j(o_t)
\]

Backward: \(\beta_T(i)=1\),
\(\beta_t(i)=\sum_j a_{ij}b_j(o_{t+1})\beta_{t+1}(j)\).
Baum–Welch re-estimates \(A,B,\pi\) from \(\gamma_t(j)=\alpha_t(j)\beta_t(j)/P(O)\)
and \(\xi_t(i,j)=\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j)/P(O)\).
CLI: `dipcatcher hmm eisner`. Research only.

## Causal risk-controlled gates (ADR-036)

Book-level size $s_t$ applied to return $r_{t+1}$ is a function of
$\{r_1,\ldots,r_t\}$ only (delay 1). Constant leverage leaves Sharpe
unchanged when rf $=0$:

\[
\mathrm{SR}(c\,r)=\mathrm{sign}(c)\,\mathrm{SR}(r),\qquad c\neq 0.
\]

Vol targeting (Moreira–Muir 2017) is *time-varying* leverage
$s_t=\mathrm{clip}(\sigma^\star/\hat\sigma_t,0,s_{\max})$. It can change
Sharpe if expected return does not scale 1:1 with vol. It cannot mint
Sharpe 5 from IC $\approx 0$. Jointly, $\sigma^\star=0.025$ and
Sharpe 5 imply excess return $\approx 12.5\%$/year ($\sim 3.4\times$,
not $10\times$). $10\times$ at 2.5% vol needs Sharpe $\sim 10$.

Fractional Kelly (Thorp), rf $=0$:

\[
f^\star=\frac{\mu}{\sigma^2},\qquad
s^{\mathrm{Kelly}}_t=\mathrm{clip}(\kappa f^\star_t,0,1).
\]

Negative $\mu$ is clipped to 0 (ADR-032). CRC size: calibrate
Angelopoulos–Bates–Malik–Jordan (2022) on losses vs a 0 bound; $\hat\lambda_t$
is the smallest expansion with CRC statistic $\le\alpha$. Then
$s^{\mathrm{CRC}}_t=\min(1,\hat\lambda_t/\widehat{\mathrm{ES}}_t)$.
ES halt: $s^{\mathrm{ES}}_t=\min(1,\mathrm{ES}^\star/\widehat{\mathrm{ES}}_t)$.
Crash: trailing $L$-bar wealth change $\le c$ (nautica $-20\%$/10d) $\to 0$.
StepM size: expanding-window Romano–Wolf (2005) on the book's returns vs 0;
if column 0 is not rejected, $s_{t+1}=0$. Drawdown halt flattens after
peak-to-trough $\le -\delta$ and stays cash; remaining-budget mode scales
by $(\delta-\mathrm{DD}_t)/\delta$.

CS challengers `tsmom` / `vme` / `krauss` are a priori or fold-fit on
`PUBLIC_FEATURES`. Champion remains public ridge. `blend_weight` 0.
Not a live P&L claim. Holdout confirmation (`dipcatcher ls confirm`)
splits already-causal date ICs at 2024-12-31 / 2025-01-02; it does
not retune and does not move the champion.

Directional (not CS-idio) Moskowitz 12–1 long-only / Antonacci GEM /
top-k long books live in `hedge_lab.directional`. Delay 1, monthly
rebalance, costs on turnover. They are not CS rankers.



## Canon-wave conventions (metrics/models canon 2026-09)

- Loss sign convention: risk functions take *losses* (positive = bad) in
  `metrics.risk_parametric`/`metrics.extremes`; `metrics.drawdown` takes
  simple *returns* and computes the drawdown path internally.
- VaR/ES quantile `alpha` in (0.5, 1) is enforced everywhere; ES is the
  mean of the tail beyond VaR (`metrics.extremes.gpd_var_es`,
  `risk_parametric.student_t_var_es` use the analytic tail formulas;
  Cornish–Fisher ES uses quadrature over the probability axis).
- Warmup semantics: `features.indicators` and `features.cycles` return
  NaN-padded outputs so a feature at index t only ever uses data <= t.
- Hawkes compensator residuals are computed in transformed time
  (`point_process.hawkes_compensator`); under the fitted model they are
  Exp(1) — use `hawkes_residuals` + `metrics.serial`/KS checks.
- `bocpd_gaussian` reports the posterior run-length distribution;
  `cp_prob[t] = P(r_t = 0)` is the exact changepoint probability, not a
  thresholded alarm.
- `eigenvalue_clip` preserves trace while zeroing noise dispersion;
  `detone_cov` removes the top eigencomponents entirely.
- `cvar_minimization` solves the Rockafellar–Uryasev LP exactly on the
  empirical scenario set (HiGHS); reported CVaR is recomputed from the
  empirical tail at the returned weights.
- Bandit `update` semantics: EXP3 requires update(arm) to match the arm
  returned by the immediately preceding select (importance weighting).

- metrics.regression OLS returns residuals + pinv(X'X); HC0-HC4 and
  Newey-West HAC covariances share the bread-meat-bread form; CUSUM uses
  standardized recursive residuals with BDE 5% lines (a=0.948*sqrt(m));
  CUSUMSQ uses the Kolmogorov asymptotic bound 1.36*sqrt(2/m); quantile
  regression solves the exact LP via HiGHS; 2SLS residuals are computed
  on ORIGINAL regressors (not fitted), Sargan J = n*R2 of resid on Z.
- models.factor_models fama_macbeth returns per-period gammas and
  Shanken-inflated SEs; bai_ng_factors runs on the RAW panel (demeaned,
  not standardized) since IC penalties assume common sigma_e^2.
- models.filters: hp_filter solves the exact sparse ridge system;
  baxter_king/corbae_ouliaris return NaN-free vs burn-in conventions
  respectively (BK NaN-pads k at both ends, CO uses full-period DFT);
  hamilton_filter residuals are MA(h-1) by construction.
- models.realized: bipower_variation is jump-robust IV; TSRV uses
  K ~ n^(2/3) price grids (price-level noise only); preaveraged_rv uses
  g=min(x,1-x) with psi1=1, psi2=1/12; lee_mykland thresholds via the
  Gumbel double-exponential law; bns_jump_test uses the Huang-Tauchen
  max(1, TPQ/BV^2) normalization.
- models.var_coint: johansen_test/vecm_fit solve the GENERALIZED
  eigenproblem |lam*S11 - S10 S00^-1 S01| = 0 via scipy.linalg.eigh
  (symmetric A, spd B) — never eigvalsh on the nonsymmetric product;
  Johansen CVs are MHM (1999) 5% asymptotic with a constant shift for
  det=1; diebold_yilmaz uses generalized (Pesaran-Shin) FEVD so it is
  ordering-invariant; spread_half_life returns inf for rho >= 1 or <= 0.
- features.liquidity: FHT and LOT map zero-return frequency to cost
  via normal quantiles; effective_tick follows Holden's incremental
  probability weighting (upward-biased on exact grids by design).
- metrics.distribution: lilliefors uses a seeded parametric bootstrap
  (exact for estimated-parameter KS); medcouple is the O(n^2) naive form;
  qn_scale uses c = 1/(sqrt2*Phi^-1(5/8)) = 2.2219.
- models.decomposition: SSA Hankelizes each rank-1 SVD component;
  ssa_forecast uses the vertical-eigenvector linear recurrence; emd
  sifting uses cubic-spline envelopes with endpoint inclusion and stops
  on monotone residue; hilbert_spectrum reports IMF1 only.
- metrics.calibration2: murphy_decomposition equals REL-RES+UNC up
  to within-bin dispersion; winkler_interval_score = width + pinball
  penalties; variogram_score is Scheuerer-Hamill p=0.5 default.
- models.mixture: t-mixture ECM uses E[ln u] = psi((nu+d)/2) -
  ln((nu+delta)/2), NOT ln E[u]; nu bounded to [3,300]; BIC counts nu.
- models.pairs: gatev SSD on normalized prices; cointegration_screen
  delegates to engle_granger; ou_optimal_bands is a grid approximation
  of the Leung-Li stopping problem, research-grade only.


## Wave 3 conventions

- `fit_markov_switching_*` return `filtered`, `smoothed` (Kim), transition matrix
  `P`, and per-state parameters; rows of `P` sum to 1 and `P[i,j] = P(s_t=j | s_{t-1}=i)`.
- FIGARCH variance uses the truncated BBM lambda recursion with
  `lambda_1 = d + phi - beta`, `lambda_k = beta*lambda_{k-1} + pi_k - phi*pi_{k-1}`.
- APARCH news function is `(|e| - gamma*e)^delta`; positive `gamma` = leverage
  asymmetry (bad news raises vol more).
- `clark_west_test` expects the forecast-difference series `f_null - f_alt`
  passed as `preds_alt`; the adjusted loss is `e_null^2 - e_alt^2 + (f_diff)^2`.
- `fluctuation_test` returns sup of rolling-window DM-type stats; GR(2010)
  asymptotic two-sided critical values ~3.18 (10%) / ~3.68 (5%).
- `hsic` uses a seeded permutation null (no parametric approximation);
  `chatterjee_xi` uses the rank statistic `1 - 3*sum|r_{i+1}-r_i|/(n^2-1)`.
- `mutual_information_knn` implements KSG estimator 1 with Chebyshev balls and
  strict `eps` marginal counts; returns `mi` in nats.
- `basel_zone` uses the fixed green<=4/yellow<=9/red>=10 table only for the
  canonical 99%/250-day case; otherwise exact binomial-tail cutoffs.
- `arellano_bond` is one-step difference GMM with block-diagonal per-period
  instrument matrices (levels y_{t-2},...,y_{t-1-maxlag}); `sargan_J` uses the
  instrument-covariance weight matrix.
- `fit_cox_ph` uses Breslow tie handling and reports Harrell's concordance.
- Lo (1991) R/S band [0.809, 1.862] on `Q/sqrt(n)` rejects short memory.
