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
\widehat{\mathrm{CRPS}}(F,y) = \sum_{k=1}^{K} L_{\tau_k}(y,q_k)\,(\tau_k-\tau_{k-1})
\]

with \(\tau_0=0\). This is an approximation, not the closed-form CRPS of a parametric law.

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

## GARCH(1,1)

\[
\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2
\]

Estimated with `arch`. Stationarity constraint \(\alpha+\beta<1\) is checked; failures are logged and the forecast is marked invalid.

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

Wealth \(W_t = \prod_{u\le t}(1+R_u)\), peak \(\mathrm{Peak}_t = \max_{u\le t} W_u\):

\[
\mathrm{DD}_t = W_t/\mathrm{Peak}_t - 1, \qquad \mathrm{MDD} = \min_t \mathrm{DD}_t
\]

MDD is \(\le 0\). Monotone increasing wealth implies MDD \(= 0\).

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

## CSCV / PBO

Combinatorial purged CV (López de Prado). PBO is the fraction of CSCV splits in which the in-sample-best trial has below-median out-of-sample performance.

## Mean-variance objective

\[
\max_w \; w^\top \alpha - \lambda_{\mathrm{risk}}\, w^\top \Sigma w - \lambda_{\mathrm{tc}}\,\mathrm{TC}(w-w_{\mathrm{prev}}) - \lambda_{\mathrm{tail}}\,\mathrm{Tail}(w)
\]

subject to configurable gross, net, name, sector, factor, turnover, and ADV constraints. Transaction cost \(\mathrm{TC}\) is a convex approximation (L1 spread/fees plus quadratic or \(\ell_2\) square-root linearization; see execution spec). Infeasible problems return diagnostics; constraints are never silently dropped.

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

- **E-values (ADR-010).** Bernoulli betting e-process on prediction-set misses versus nominal \(\alpha\). \(E_t=\prod_{s\le t}e_s\) is a nonnegative martingale under the null; Ville's inequality uses threshold \(1/0.05=20\) with no peeking correction.
- **Jackknife+ (ADR-011).** Leave-one-out residual conformal wrapper. Finite-sample coverage is \(\ge 1-2\alpha\), not \(1-\alpha\).
- **Conformal Risk Control (ADR-012).** Smallest expansion \(\hat\lambda\) such that \((n\widehat L_n(\lambda)+B)/(n+1)\le\alpha\) for monotone 0-1 VaR-hit loss.
- **Weighted split CQR (ADR-013).** Tibshirani–Barber–Candès–Ramdas likelihood-ratio weights on a PIT-safe vol covariate. Lab scores remain coverage and width.
- **Interval position caps (ADR-014).** \(\mathrm{cap}=\bar w\cdot w_{\mathrm{ref}}/(w_{\mathrm{ref}}+\mathrm{width})\cdot d_{\mathrm{ref}}/(d_{\mathrm{ref}}+\max(0,-\ell))\). Binding fraction and mean cap only; no Sharpe.
- **Quantile Thompson (ADR-015).** Shared linear pinball models on a \(\tau\)-grid; posterior draw then top-\(k\). Reward is the scientific residual, not a portfolio path.
