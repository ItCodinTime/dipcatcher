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
ordered, date-level equal-weight mean of finite `ret_1` values across assets; panel
rows are never concatenated into a synthetic time series. A production asset-specific
GARCH requires a separate per-asset fit and artifact namespace. For numerical stability
`arch` receives \(100r_t\), so its conditional variance is in percent-squared units and
is divided by \(10000\) at the API boundary. QLIKE is evaluated on decimal variance,
not sigma.

For symmetric GARCH(\(p,q\)):

\[
\sigma_t^2 = \omega + \sum_{i=1}^{p}\alpha_i\varepsilon_{t-i}^2
              + \sum_{j=1}^{q}\beta_j\sigma_{t-j}^2.
\]

GJR adds \(\sum_i\gamma_i I(\varepsilon_{t-i}<0)\varepsilon_{t-i}^2\), while EGARCH
models log variance and handles positivity in log space. The supported innovation
families are Gaussian, standardized Student-t, and skewed Student-t.

`GARCHVol.forecast(horizon=h)` returns a deterministic origin-indexed path of
per-future-bar decimal variances and sigmas. `cumulative_variance` is the sum of
those conditional variances for comparison with the close-to-close forward
realized-variance label. Predictive quantiles and PIT values use the fitted
innovation distribution and are bounded to \([0,1]\) for valid inputs.

The fit gate checks finite parameters, finite positive conditional volatility,
optimizer convergence, and full-order persistence. For GARCH and GJR the
persistence diagnostic is \(\sum_i\alpha_i+\sum_j\beta_j+\frac12\sum_i\gamma_i\);
EGARCH is not subjected to this symmetric persistence formula. Short, constant,
non-converged, non-finite, or non-stationary fits fail closed to a sample-sigma
fallback with an explicit diagnostic reason. This fallback is a safety mechanism,
not evidence of model superiority.

## HAR-RV (Corsi)

\[
\mathrm{RV}_{t+1} = \beta_0 + \beta_d \mathrm{RV}_t + \beta_w \overline{\mathrm{RV}}_{t-4:t} + \beta_m \overline{\mathrm{RV}}_{t-21:t} + \varepsilon_{t+1}
\]

When daily RV is unavailable, close-to-close squared log return proxies RV. Log-RV regression is used when `volatility.har_log: true`.

## Ledoit–Wolf shrinkage

sklearn `LedoitWolf` shrinks the sample covariance \(S\) toward \(\mu I\):

\[
\Sigma = (1-\delta)S + \delta \mu I, \qquad \mu = \mathrm{tr}(S)/N
\]

This is the 2004 linear formula, not nonlinear shrinkage (Ledoit–Wolf 2017). Nonlinear shrinkage is deferred.

## DCC(1,1)

Two-stage Engle (2002). Stage 1: univariate GARCH via `arch`. Standardized residuals \(z_t\). Stage 2:

\[
Q_t = (1-a-b)\bar Q + a z_{t-1}z_{t-1}^\top + b Q_{t-1}
\]

\[
R_t = \mathrm{diag}(Q_t)^{-1/2} Q_t \mathrm{diag}(Q_t)^{-1/2}, \qquad H_t = D_t R_t D_t
\]

Parameters \(a,b>0\), \(a+b<1\) are estimated by QML on the correlation likelihood. Implemented in-house (ADR-004).

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

Starting from \(W_0=1\), wealth \(W_t = \prod_{u=1}^{t}(1+R_u)\) and peak \(\mathrm{Peak}_t = \max_{0\le u\le t} W_u\):

\[
\mathrm{DD}_t = W_t/\mathrm{Peak}_t - 1, \qquad \mathrm{MDD} = \min_t \mathrm{DD}_t
\]

MDD is \(\le 0\). A first-period loss is measured against the initial capital, so \(R_1=-0.2\) gives \(\mathrm{DD}_1=-0.2\). Monotone increasing wealth implies MDD \(= 0\). If any return or drawdown observation is non-finite, the metric is NaN rather than a minimum over a silently truncated finite subset.

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
