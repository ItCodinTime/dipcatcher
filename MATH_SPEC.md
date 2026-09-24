

## Canon wave 4 notation additions

### Nonlinear filters
- EKF: Jacobians F_t = df/dx, H_t = dh/dx; predict/update identical to Kalman with linearized matrices.
- UKF sigma points: chi_0 = x, chi_i = x +- (sqrt((n+lam)P))_i; weights w_0 = lam/(n+lam) (mean), w_c0 = w_0 + (1 - alpha^2 + beta) (cov).
- Bootstrap PF: p(x_t | y_1:t) ~ sum_i w_t^i delta(x_t - x_t^i); effective sample size N_eff = 1 / sum_i (w^i)^2; systematic resampling.

### ES backtesting
- Acerbi-Szekely (Z_2): Z_2 = n*(mean_t[x_t * 1{x_t > v_t} / e_t] - (1 - alpha)) / (1 - alpha); H_0 E[x | x > VaR] = ES.
- McNeil-Frey: exceedance residuals e_t = (x_t - v_t)/e_t over violations; bootstrap null by resampling residuals.
- Sign convention: losses positive; ES >= VaR enforced; tests fail closed on violation.

### Spectral
- Welch: periodograms averaged over K overlapping segments with window w; density pxx(f) = (1/K) sum_k |FFT(w x_k)|^2 / (f_s * U).
- Daniell: uniform moving-average smoothing of raw periodogram over m ordinates.
- Thomson multitaper: DPSS tapers v_k, eigencoefficients y_k(f) = sum_t v_k(t) x_t e^{-i2pi ft}, adaptive weights d_k^2 = lam_k / (1 - lam_k) * (1/b) where b estimated iteratively.
- Coherence: C_xy(f) = |S_xy(f)|^2 / (S_xx(f) S_yy(f)) in [0, 1].

### Density forecast / PIT
- u_t = F_t(y_t); correct density iff u_t iid U(0,1).
- Berkowitz: z_t = Phi^{-1}(u_t) should be iid N(0,1); LR test on (mu, sigma^2, AR(1) rho).
- PIT autocorr: portmanteau on (u_t - 0.5) and (u_t - 0.5)^2.

### Nowcasting
- MIDAS: y_t = beta0 + beta1 * sum_j w_j(theta) x_{t-j/K} + e_t; beta weights w_j = f(j/K; theta1, theta2) / sum f.
- Almon: w_j = sum_{r=0}^{P} a_r j^r, P small.
- Bridge: forecast y via equation on monthly aggregates of the daily indicator.

### Panel cointegration
- Kao ADF: residual ADF with panel-corrected variance (S_bartlett / S_{eta,tau}); H_0 no cointegration.
- Pedroni: 7 statistics — panel v/rho/t (nonparametric + ADF) and group rho/t; standardized via tabled moments.

### Momentum
- JT: portfolio = top decile - bottom decile of prior J-month return, held K months; W/L legs intersect lags.
- 52wk-high: rank on P_t / max(P over 52wk); doc: anchor-nearness as signal.
- Residual momentum: residual return after FF3 regression / sigma_residual (Sharpe of abnormal return).
- TSMOM: sign(r_{t-12:t-1}) scaled by 1/ex-ante vol; position ~ sign * (target vol / realized vol).

### Hedging
- Ederington: h* = Cov(dS, dF) / Var(dF) (OLS slope of spot on futures changes); EHR = 1 - Var(dS - h* dF)/Var(dS).
- Basis risk: Var(S_t - h F_t) vs unhedged Var(S_t); variance reduction = effectiveness.

### CCM / S-map
- Shadow manifold M_x = {x_t, x_{t-tau}, ..., x_t-(E-1)tau}; x -> y convergent iff M_x library predicts y values with skill rising in library size.
- S-map weights: w_i = exp(-theta * ||x_i - x*|| / scale); theta=0 reduces to OLS (linear).

### Diffusion index
- Factors F_t = first k PC scores of standardized panel X (T x N); forecast y_{t+h} = a + sum_j b_j F_{t,j} + sum_l c_l y_{t+1-l} + e.

### Rough volatility
- log-variance increments: m(q, Delta) = <|log sigma_{t+Delta} - log sigma_t|^q> ~ Delta^{qH}; regress log m on log Delta over q grid; slope = H.
- Variance curve: <(log sigma_{t+Delta} - log sigma_t)^2> = 2 nu^2 Delta^{2H} (stationary Gaussian); intercept gives nu.
- Fractional OU: dX = -kappa X dt + sigma dB^H; Euler-Maruyama on circulant/Cholesky-simulated fGn.

### Options
- d1 = (ln(S/K) + (r + sigma^2/2)T) / (sigma sqrt(T)); call = S N(d1) - K e^{-rT} N(d2).
- Greeks: delta_c = N(d1), delta_p = N(d1) - 1; gamma = n(d1)/(S sigma sqrt T); vega = S n(d1) sqrt(T) (per unit vol); theta, rho standard.
- Implied vol: Brent on [1e-9, 5] bounded by |S - K e^{-rT}| <= price <= S (call), fail-closed outside arbitrage bounds.


## Canon wave 5 notation additions

### SABR
- sigma_B(K) = alpha / (fK)^{(1-b)/2} / [1 + (1-b)^2/24 ln^2(f/K) + (1-b)^4/1920 ln^4(f/K)] * (z/x(z)) * C(t)
  with z = (nu/alpha)(fK)^{(1-b)/2} ln(f/K), x(z) = ln[(sqrt(1-2rho z+z^2)+z-rho)/(1-rho)],
  C(t) = 1 + t [ (1-b)^2/24 * alpha^2/(fK)^{1-b} + rho b nu alpha/(4 (fK)^{(1-b)/2}) + (2-3 rho^2)/24 nu^2 ].
- ATM cubic in alpha solved by brentq; calibration fixes beta (weakly identified with rho, nu).

### Merton jump-diffusion
- call = sum_n Pois(n; lam T) * BSM(S, K, T, r - lam kappa + n(mu_j + s_j^2/2)/T, sqrt(sigma^2 + n s_j^2/T)),
  kappa = exp(mu_j + s_j^2/2) - 1.
- Kou: kappa = p eta1/(eta1-1) + (1-p) eta2/(eta2+1) - 1 (eta1 > 1 for finite mean).

### SETAR / Hansen threshold
- SSR(gamma) = SSR_1 + SSR_2 over regime split at y_{t-d} <= gamma; F(gamma) = n (S0 - S(g))/S(g);
  H0 linear AR(p); bootstrap recurses the H0-fitted AR with resampled residuals.

### Trade signing
- Tick rule: sign of last nonzero price change; Lee-Ready quote rule with mid-tick fallback.
- BVC: V_buy_i = V_i Phi(Delta P_i / sigma_DeltaP).

### Avellaneda-Stoikov
- r(s,q,t) = s - q gamma sigma^2 (T-t); spread = gamma sigma^2 (T-t) + (2/gamma) ln(1 + gamma/kappa);
  lambda(delta) = A e^{-kappa delta} calibrated by ln-rate vs depth regression.

### Information share
- VECM Delta p_t = alpha ect_{t-1} + Gamma Delta p_{t-1} + eps; psi = alpha_perp normalized to sum 1;
  IS_j = (psi' F)_j^2 / (psi' Omega psi), F = chol(Omega); bounds via both Cholesky orderings.
- Gonzalo-Granger weights = alpha_perp / sum(alpha_perp) — linear, not convex, combination.

### Bayesian NIG
- V_n = (V0^{-1} + X'X)^{-1}; a_n = a0 + n/2; b_n = b0 + 0.5(y'y + b0'V0^{-1}b0 - b_n'V_n^{-1}b_n);
  predictive Student-t(df=2 a_n, mean=x'b_n, scale^2=(b_n/a_n)(1+x'V_n x));
  log-ML: a0 ln b0 - a_n ln b_n + 0.5 ln|V_n|/|V0| + lgamma diff - n/2 ln pi.
- BMA over all 2^p subsets with Zellner g-prior (default g = max(n, k^2)); PIP = sum probs over subsets containing column.

### Robust covariance
- OGK: gamma_ij = (s(y_i+y_j)^2 - s(y_i-y_j)^2)/4 with MAD scale s; orthogonalize by eigh; rescale.
- FastMCD: h-subset C-steps minimizing log-det; classical reweight at chi2(p, 0.975).
- Stahel-Donoho: o_i = max_a |x_i' a - med(X'a)| / MAD(X'a) over random unit directions.

### MF-DFA
- F_q(s) = [ (1/2N_s) sum_v F_v(s)^q ]^{1/q}; F_0 = exp(mean ln F); F_q ~ s^{h(q)};
  tau(q) = q h(q) - 1; alpha = dtau/dq; f(alpha) = q alpha - tau; width = alpha_max - alpha_min.

### MODWT
- Circular filters h~ = h/sqrt(2), QMF g_l = (-1)^{l+1} h_{L-1-l}; W_j and V_J length n;
  energy: ||x||^2 = sum_j ||W_j||^2 + ||V_J||^2; wavelet variance per scale; MRA additive.

### Term structure
- NS: y(m) = b0 + b1 (1-e^{-x})/x + b2[(1-e^{-x})/x - e^{-x}], x = m/lam.
- Svensson adds b3 with second (1-e^{-x2})/x2 - e^{-x2} hump.
- Diebold-Li: per-date OLS betas on fixed-lam loadings; AR(1) factor dynamics; curve forecast = beta_h' L.

### ACD
- x_i = psi_i eps_i; psi_i = omega + alpha x_{i-1} + beta psi_{i-1}; eps Exp(1) or mean-1 Weibull
  (scale Gamma(1+1/gamma)); QMLE with alpha+beta<1; diagnostics on x/psi (LB, mean-1, kurtosis).

### Directional accuracy
- PT = sqrt(n)(P_hat - P*)/sqrt(P*(1-P*)) ~ N(0,1), P* = p_y p_f + (1-p_y)(1-p_f);
  directional AUC via rank-based Mann-Whitney form.

## Wave 6 mathematical conventions

- `gas.py`: f_{t+1} = w + A u_t + B f_t; u_t = (dlogL/df)/sqrt(I) for
  ``inv_sqrt``, u_t = 2 f (dlogL/df) for ``unit`` scaling. Student-t
  density is standardized to variance f (nu > 4 required).
- `caviar.py`: all four EM2004 dynamics use q_{t+1} = b0 + b1 q_t +
  b2 |r_t| (SAV); asymmetric slope splits r^+, r^-; adaptive uses the
  GARCH-style recursion q_{t+1} = q_t + b (1{ r_t < q_t } - tau) /
  (1 + exp(g (r_t - q_t))) ; indirect GARCH maps a GARCH variance path
  to quantiles.
- `garch_midas.py`: sigma2_t = tau_t * g_t; tau_t = exp(m + theta *
  sum_k phi_k(omega) RV_{t-k}); g_t = (1-a-b) + a r^2_{t-1}/tau_{t-1}
  + b g_{t-1}.
- `gmm_est.py`: two-step Hansen estimator; S = Bartlett NW long-run
  variance (Andrews data-driven bandwidth default); J = T g' S^{-1} g.
- `arma.py`: CSS residuals e_t = y_t - phi' y_lags - theta' e_lags,
  e_t = 0 for t <= max(p,q); Hannan-Rissanen stage-1 AR order =
  ceil(min(p+q+5, log(n)^1.5, n/3)); AIC = n_eff ln sigma2 + 2k.
- `poet.py`: Sigma = Lambda Lambda' + Sigma_u; off-diagonal residual
  correlations hard-thresholded at c = sqrt(ln p / T); Sigma_u
  eigenvalue-clipped to PSD.
- `glasso.py`: FHT block coordinate descent; inner lasso by cyclic
  coordinate descent (soft-threshold, tol 1e-7); theta blocks
  recovered via theta22 = 1/(s22+rho-w12'b).
- `functional.py`: eigenproblem symmetrized as W^{1/2} C W^{1/2}
  (trapezoid weights); eigenfunctions phi = W^{-1/2} psi.
- `vol_eval.py`: QLIKE uses ratio form x/f - ln(x/f) - 1; MZ joint
  test is a Wald chi2(2) on (a, b) = (0, 1).


## Wave 7 mathematical conventions

- `egarch.py`: EGARCH ln sig2 recursion uses z = (r - mu)/sig with
  E|z| = sqrt(2/pi); GJR stationarity enforced as a + g/2 + b < 1.
- `dcc.py`: stage-1 scalar GARCH(1,1) per series; stage-2 targets
  Q_t = (1-a-b)Qbar + a zz' + b Q_{t-1}; R_t = D_t^{-1/2} Q_t D_t^{-1/2}.
- `stoch_vol.py`: z = ln y^2 - E[ln chi^2_1] (E = -1.2704, R = pi^2/2);
  phi constrained by tanh; stationary init via discrete Lyapunov.
- `ordered.py`: cutpoints c_j = c_1 + sum exp(delta_i); no intercept
  column in x (absorbed by cutpoints).
- `kmv.py`: solves E = VN(d1) - D e^{-rT} N(d2) and
  sE = (V/E) N(d1) sV jointly; DD = ln(V/D) / (sV sqrt(T)).
- `purged_cv.py`: purge drops train samples whose [i, t1_i) label span
  intersects the test block; embargo drops a fixed fraction of n after
  each test block.
- `feature_select.py`: FCD mRMR with |corr| relevance/redundancy.


## Wave 8 mathematical conventions

- `ets.py`: SES level l_t = a y_t + (1-a) l_{t-1}; Holt adds trend
  b_t = beta (l_t - l_{t-1}) + (1-beta) phi b_{t-1} with damped factor
  phi in [0.8, 1]; multi-step damping sum_{i=1..h} phi^i (= h when phi=1).
  Holt-Winters additive s_t = gamma (y_t - l_{t-1} - phi b_{t-1}) +
  (1-gamma) s_{t-m}; multiplicative uses ratios y_t / s_{t-m}. Smoothing
  params fit by bounded L-BFGS-B on one-step SSE; states seeded from the
  first one/two seasons; AIC = n ln(SSE/n) + 2k.
- `theta.py`: theta line Z_t(theta) = theta y_t + (1-theta)(a + b t) with
  (a, b) the OLS trend. Reconstruction weight 1/theta on the theta line and
  1 - 1/theta on the trend line (equal 1/2 weights at theta=2). Forecast =
  (1/theta) SES-extrapolation(Z) + (1 - 1/theta)(a + b(n+h-1)); drift is
  ~ b/2 (Hyndman-Billah).
- `croston.py`: SES on non-zero sizes z and inter-arrival intervals p;
  rate = z_hat / p_hat. SBA scales by (1 - alpha/2). TSB smooths demand
  probability every period: prob_t = beta 1{y_t>0} + (1-beta) prob_{t-1},
  size updated only on occurrences; rate = prob_hat * z_hat.
- `robust_location.py`: Hodges-Lehmann = median of Walsh averages
  {(x_i + x_j)/2 : i <= j}; two-sample = median{x_i - y_j}. Siegel
  repeated median slope = median_i median_{j != i} (y_j - y_i)/(x_j - x_i),
  intercept = median_i (y_i - slope x_i) (50% breakdown).
- `expectile.py`: tau-expectile minimises E[w_tau(y-mu)(y-mu)^2],
  w_tau = tau if residual > 0 else 1 - tau; solved by IRLS (WLS with
  sqrt-weights) which converges on the convex objective. EVaR = tau-expectile
  of losses, coherent for tau >= 1/2.
- `spectral_risk.py`: M_phi = sum_i w_i L_(i) over ascending losses, with
  band weights w_i = integral_{(i-1)/n}^{i/n} phi(p) dp, phi non-negative,
  non-decreasing, sum 1. Exponential phi(p) = k e^{-k(1-p)}/(1-e^{-k});
  power phi(p) = gamma p^{gamma-1}; ES is the uniform tail spectrum on
  [alpha, 1] scaled by 1/(1-alpha).

## Wave 10 mathematical conventions

- `entropic_risk.py`: rho_theta(L) = (1/theta) log E[e^{theta L}] (log-sum-exp
  stabilised). EVaR_{1-alpha} = inf_{z>0} (1/z) log(E[e^{zL}]/(1-alpha)),
  minimised over log z; satisfies EVaR >= CVaR >= VaR.
- `perf_ratios.py`: ASR = SR[1 + (S/6)SR - ((K-3)/24)SR^2] (Pezier-White);
  M^2 = rf + SR * sigma_benchmark; Rachev = E[x | x >= Q_{1-beta}] /
  (-E[x | x <= Q_alpha]); gain-to-pain = sum(x)/sum(max(-x,0)); UPR =
  E[(x-mar)_+] / sqrt(E[((x-mar)_-)^2]).
- `stable.py`: Chambers-Mallows-Stuck sampler with U~Unif(-pi/2,pi/2), W~Exp(1);
  zeta=-beta tan(pi alpha/2), xi=atan(-zeta)/alpha. ECF fit regresses
  log(-log|phi(t)|) on log|t|: slope=alpha, intercept=alpha log c; loc=median.
- `resampled.py`: for s=1..S draw N(mu,cov) of length n_obs, re-estimate
  (mu_s, cov_s), solve simplex-constrained max-Sharpe / min-variance (SLSQP),
  average the weights and renormalise (Michaud resampled efficiency).

## Wave 20 mathematical conventions

- `concordance.py`: from concordant C, discordant D, ties Tx/Ty: gamma=(C-D)/(C+D);
  Somers' D_{y|x}=(C-D)/(C+D+Ty); tau-b=(C-D)/sqrt((C+D+Tx)(C+D+Ty)); c-index =
  (#{s_pos>s_neg}+0.5 #ties)/(n_pos n_neg) (= AUC).
- `calibration_tests.py`: Spiegelhalter Z = sum (y-p)(1-2p)/sqrt(sum (1-2p)^2
  p(1-p)) ~ N(0,1) under calibration; insensitive to symmetric additive shifts,
  sensitive to over/under-confidence.
- `gam.py`: backfitting y=intercept+sum_j f_j(x_j); each f_j refit from the
  partial residual with a Gaussian local-linear smoother (Silverman bandwidth)
  and centred to zero mean; iterate to convergence.
## Wave 21 mathematical conventions

- `forecast_accuracy.py`: U1=sqrt(MSE)/(sqrt(mean a^2)+sqrt(mean f^2)); U2=
  sqrt(sum(f-a)^2)/sqrt(sum(a_t-a_{t-1})^2); MASE=mean|a-f|/mean|a_t-a_{t-m}|.
- `whitening.py`: Sigma=U Lambda U'; PCA W=Lambda^{-1/2}U'; ZCA W=U Lambda^{-1/2}
  U' (symmetric); whitened Cov = I.
- `efficient_frontier.py`: A=1'S^{-1}1, B=1'S^{-1}mu, C=mu'S^{-1}mu, D=AC-B^2;
  min-var w=S^{-1}1/A; frontier w=g+h m, var=(A m^2-2B m+C)/D; tangency=
  S^{-1}(mu-rf)/1'S^{-1}(mu-rf).
- `utility.py`: u(w)=(w^{1-gamma}-1)/(1-gamma) (log for gamma=1); CE gross =
  (mean gross^{1-gamma})^{1/(1-gamma)} (geometric mean for gamma=1); CE<mean for
  gamma>0.
