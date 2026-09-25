

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


## Wave 9 mathematical conventions

- `skew_normal.py`: f(x) = (2/omega) phi(z) Phi(alpha z), z=(x-xi)/omega;
  CDF = Phi(z) - 2 T(z, alpha) (Owen's T). MoM inverts skewness g1 to
  delta = alpha/sqrt(1+alpha^2) with |g1| capped at 0.995; omega, xi match
  the first two moments.
- `johnson_su.py`: z = gamma + delta asinh((x-xi)/lambda). Slifker-Shapiro
  reads quantiles at z=+/-z0, +/-3z0; delta = 2 z0 / acosh((m/p+n/p)/2),
  gamma = delta asinh((n/p-m/p)/(2 sqrt(mn/p^2-1))); lambda and xi are
  recovered exactly from the standardised inner spread p = lambda[sinh((z0-g)/d)
  - sinh((-z0-g)/d)].
- `edgeworth.py`: Gram-Charlier A f = phi(z)/sigma [1 + (S/6)He_3 + (K/24)He_4],
  F = Phi(z) - phi(z)[(S/6)He_2 + (K/24)He_3], He_2=z^2-1, He_3=z^3-3z,
  He_4=z^4-6z^2+3; integrates to 1 for any (S,K) but may go negative
  (validity checked on a grid).
- `tukey_gh.py`: x = A + B (e^{gZ}-1)/g e^{hZ^2/2} (limit B Z e^{hZ^2/2} as
  g->0). Monotone in Z for h>=0 so CDF/PDF follow by inversion and
  f(x)=phi(Z)/(dx/dZ). Hoaglin fit: g = (1/Z) ln(UHS/LHS) per symmetric
  quantile pair; regress ln[FS g/(2 sinh gZ)] on Z^2/2 for (ln B, h).
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

## Wave 11 mathematical conventions

- `skew_t.py`: standardised Hansen skew-t with c=Gamma((nu+1)/2)/(sqrt(pi(nu-2))
  Gamma(nu/2)), a=4 lambda c (nu-2)/(nu-1), b=sqrt(1+3 lambda^2-a^2); density
  splices (bz+a)/(1-/+lambda); CDF via scaled Student-t of each branch; ML fit.
- `ar_estimation.py`: Yule-Walker solves R phi = r; Levinson-Durbin recursion
  k_m=(r_m - sum phi_j r_{m-j})/E_{m-1}, E_m=E_{m-1}(1-k_m^2); Burg minimises
  forward+backward errors, phi=-a[1:], reflection coeffs per order.
- `empirical_copula.py`: pseudo-obs U=rank/(n+1); C_n(u)=(1/n) sum_i prod_j
  1{U_ij<=u_j}; lower/upper tail dependence C(t,t)/t and (1-2t+C)/(1-t).
- `dcca.py`: integrate profiles, detrend length-(s+1) overlapping windows by an
  order-1 polynomial, F2_DCCA(s)=mean box cross-cov; rho_DCCA=F2_DCCA/
  sqrt(F2_DFA_x F2_DFA_y) in [-1,1]; exponent = slope of log|F_DCCA| vs log s.
- `elliptical.py`: multivariate-t EM with weights w_i=(nu+d)/(nu+maha_i),
  mu=sum w_i x_i/sum w_i, Sigma=(1/n) sum w_i (x-mu)(x-mu)'; nu by profile-
  likelihood 1-D search.
## Wave 13 mathematical conventions

- `bachelier.py`: d=(F-K)/(sigma sqrt(T)); call=df[(F-K)Phi(d)+sigma sqrt(T)phi(d)];
  ATM=df sigma sqrt(T/(2 pi)); implied normal vol by brentq.
- `short_rate.py`: Vasicek/CIR affine bonds P=A(tau) e^{-B(tau) r}; Vasicek exact
  AR(1) discretisation r_{t+1}=r_t e^{-kappa dt}+theta(1-e^{-kappa dt})+sd Z, OLS
  calibration recovers (kappa,theta,sigma); CIR B=2(e^{g tau}-1)/((g+kappa)
  (e^{g tau}-1)+2g), g=sqrt(kappa^2+2 sigma^2), full-truncation Euler sim.
- `bond_analytics.py`: continuous compounding P=sum c_i e^{-y t_i}; Macaulay
  D=(1/P)sum t_i c_i e^{-y t_i} (=modified under continuous comp.); convexity
  (1/P)sum t_i^2 c_i e^{-y t_i}; YTM by brentq; DV01=D*P*1e-4.
- `variance_swap.py`: DDKZ fair strike K_var=(2/T)e^{rT} sum (dK_i/K_i^2) Q(K_i)
  - (1/T)(F/K0-1)^2 with OTM puts below K0 and calls at/above; equals sigma^2
  under flat Black-Scholes vol.
## Wave 12 mathematical conventions

- `nig_vg.py`: NIG f=(alpha delta/pi) exp(delta gamma + beta(x-mu)) K_1(alpha g)/g,
  g=sqrt(delta^2+(x-mu)^2), gamma=sqrt(alpha^2-beta^2); closed-form MoM inverts
  (mean,var,skew,exkurt) with bar_beta^2 = r/(3-4r), r=s^2/k. VG uses the
  Madan-Carr-Chang density with K_{1/nu-1/2}; both sampled as normal
  mean-variance mixtures (inverse-Gaussian / gamma subordinators).
- `mic.py`: MIC = max over grids with n_x n_y <= n^0.6 of I(X;Y)/log(min(n_x,n_y))
  using equi-frequency (quantile) bins; value in [0,1].
- `archimedean_extra.py`: Frank C=-1/theta log(1+ (e^{-theta u}-1)(e^{-theta v}-1)/
  (e^{-theta}-1)), tau=1-4/theta+4 D_1(theta)/theta (Debye), fit by tau inversion;
  Joe C=1-(a+b-ab)^{1/theta}, a=(1-u)^theta, density c=S^{1/theta-2}
  ((1-u)(1-v))^{theta-1}(theta-1+S), fit by ML. Both sampled by conditional
  inversion of dC/du.
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

## Wave 14 mathematical conventions

- `local_projection.py`: for each horizon h regress y_{t+h}=a_h+beta_h x_t+
  controls+e; IRF(h)=beta_h; Newey-West HAC variance with bandwidth h+1.
- `empirical_likelihood.py`: W(mu0)=2 sum log(1+lambda(x_i-mu0)) with lambda
  solving sum (x_i-mu0)/(1+lambda(x_i-mu0))=0 on the feasible interval; W ~
  chi2_1; CI by inverting W(mu0)<=chi2_{1,level}.
- `kde.py`: fhat=(1/nh) sum phi((x-x_i)/h); Silverman h=0.9 min(sd,IQR/1.34)
  n^{-1/5}; Scott h=1.059 sd n^{-1/5}; LSCV minimises
  (1/n^2) sum (1/(2h sqrt pi)) e^{-d^2/4} - (2/(n(n-1)h)) sum_{i!=j} phi(d).
- `lowess.py`: local linear fit over the frac*n nearest points with tricube
  weights (1-|u|^3)^3; robustness iterations reweight by Tukey bisquare
  (1-(r/6s)^2)^2, s=median|resid|.
## Optional scorecard families (benches_extra)

- `complexity` / `roughness` / `serial_randomness` benches average per-asset
  proper-score diagnostics over the SYNTHETIC panel's `ret_1` series (assets with
  >= 64/128 observations), reusing the entropy/fractal/serial metric modules; the
  resulting family blobs contain only finite floats with no sharpe/sortino/
  calmar/pnl/nav key tokens, so the verify.py scorecard honesty gates pass.
## Wave 15 mathematical conventions

- `fgls_ar1.py`: estimate rho from OLS residuals (rho=sum r_t r_{t-1}/sum r_{t-1}^2);
  Cochrane-Orcutt quasi-differences y_t-rho y_{t-1} (drops obs 1); Prais-Winsten
  keeps obs 1 scaled by sqrt(1-rho^2); iterate to convergence.
- `twosample.py`: KS D=max|F_x-F_y|, p via Kolmogorov asymptotic with the
  Stephens small-sample correction on sqrt(ne) D, ne=n_x n_y/(n_x+n_y); energy
  distance E=2 mean|x_i-y_j| - mean|x_i-x_j| - mean|y_i-y_j|; permutation
  p-values (count of shuffled statistics >= observed, +1 smoothing).
## Wave 16 mathematical conventions

- `american_baw.py`: b=r-q; q2/q1 = (-(N-1) +/- sqrt((N-1)^2+4M/Kc))/2,
  M=2r/sigma^2, N=2b/sigma^2, Kc=1-e^{-rT}; critical price by root finding;
  C_A = C_E + A2 (S/S*)^{q2} for S<S*, else S-K (puts analogous with q1).
- `johnson_sb.py`: SB z=gamma+delta log(u/(1-u)), u=(x-xi)/lambda on (xi,xi+lam);
  SL z=gamma+delta log(x-xi); fit fixes support then regresses Phi^{-1}(rank)
  on the transform (slope=delta, intercept=gamma).
- `carr_madan.py`: psi(v)=e^{-rT} phi(v-(alpha+1)i)/(alpha^2+alpha-v^2+
  i(2alpha+1)v); C(k)=e^{-alpha k}/pi Re int e^{-ivk} psi dv via FFT with
  log-strike spacing lambda=2pi/(N eta) and Simpson weights.
## Wave 19 mathematical conventions

- `random_projection.py`: JL min dim k = ceil(4 ln n /(eps^2/2 - eps^3/3));
  Gaussian R ~ N(0,1/k); Achlioptas sparse R entries sqrt(s/k){+1,0,-1} at
  probs {1/2s, 1-1/s, 1/2s}, s=1/density.
- `spectral_clustering.py`: W=exp(-||xi-xj||^2/(2 sigma^2)) (median-heuristic
  sigma, zero diagonal); L_sym=I-D^{-1/2}WD^{-1/2}; embed with k smallest
  eigenvectors, row-normalise, k-means.
- `cluster_validity.py`: silhouette s=(b-a)/max(a,b); Calinski-Harabasz
  (B/(k-1))/(W/(n-k)); Davies-Bouldin mean_i max_{j!=i}(s_i+s_j)/d_ij.
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
