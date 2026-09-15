# Artificial Hedge · Dipcatcher scientific lab

Dipcatcher is a **hedge research laboratory**. It measures forecast quality with proper scoring rules. It does not exist to manufacture Sharpe ratios.

## Families (`quant research` / `quant lab`)

| Family | Scientific scores |
|---|---|
| Ranking | Date-level IC / RankIC, HAC t, decile monotonicity |
| Alpha | Holdout MSE vs historical mean, Pearson IC |
| Volatility | QLIKE, Diebold–Mariano |
| Distribution | Pinball, CRPS, interval coverage, crossing, PIT KS (raw Gaussian + vol-scaled Gaussian/t); 1d and 5d scored as separate keys |
| Regime | HMM AIC/BIC, holdout average log-likelihood |
| Tail | Historical VaR/ES; headline Kupiec is vol-scaled; unscaled kept as diagnostic |
| Drawdown | Brier, log-loss, ECE vs base rate |
| Liquidity | Amihud correlation, Almgren–Chriss vs TWAP shortfall |
| Reinforcement | LinUCB top-k on **public** features; regret vs planted oracle; advantage vs uniform |
| Conformal | Operational CQR/ACI/Mondrian wrap scaled (t) bands; `cqr_raw`/`aci_raw` show misspecification repair; `|Y|` slice is not an X-validity claim |
| E-values | Anytime-valid miss e-process on ACI sets (Ville); coverage, \(E_n\), ever-cross |
| Jackknife+ | Leave-one-out conformal coverage and width; finite-sample floor \(1-2\alpha\) (bound check, not a Kupiec null) |
| CRC | CRC on scaled (t) VaR bounds (same wrappee as two-sided); homoskedastic Gaussian is diagnostic |
| Weighted conformal | Likelihood-ratio split CQR coverage and width vs exchangeable CQR |
| Interval risk | Equal-weight \(1/n\) per date, then interval caps; bind_wide vs bind_tight (no P&L) |
| Quantile bandit | Quantile Thompson on **public** features; regret vs residual oracle |

## What is not a lab headline

Sharpe, PSR, DSR, CSCV-PBO, and simulated P&L stay out of the research notebook. Those belong to optional execution backtests, not to the scientific benches.

## SYNTHETIC oracle

`planted_signal` at close \(t\) is a labeled residual oracle for \(r_{t+1}\). Recovering date-level IC is a **correctness** test of ranking only (H1), never a public-feature fit column. Alpha, drawdown, LinUCB, and Quantile Thompson train on `PUBLIC_FEATURES`. The H6/H14 headline is whether the bandit beats a **static public ridge top-k** on the same features and dates; vs-uniform and vs-planted (or y-greedy) regret stay diagnostics. Tag every such number SYNTHETIC.

## Hypothesis families

BH-FDR is **not** one table. **Calibration** (H4, H7, H8, H9, H11, H12): fail-to-reject is success. **Discovery** (H1–H3, H5, H6, H13, H14): reject is a finding. **H10 Jackknife+** is a coverage-floor boolean vs \(1-2\alpha\), not either family.
