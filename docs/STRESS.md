# Stress and scenario engine

Research simulation only. The engine replays cited crisis scalars and a
small public-domain rate/FX extract, draws synthetic scenarios, searches a
Mahalanobis ball for a worst plausible loss, and backtests VaR and expected
shortfall. It does not submit orders, connect to a broker, or rewrite sealed
receipts.

Nothing in a stress report is a live-trading result. Synthetic blocks are
labelled `SYNTHETIC` and are calibration checks, not market evidence.
Headline keys `sharpe`, `sortino`, `calmar`, `pnl`, and `nav` are rejected.

## What a loss number is

Each crisis is reported as two losses that are not added:

| Column | Meaning |
|---|---|
| Published factor shock | Primary scalars from the catalog, mapped through the strategy. Context scalars are printed and not applied. |
| Public-domain path | H.15 yield paths (modified duration) or the H.10 EURCHF noon path (simple drawdown). |

A `return` position multiplies weight by a simple return. A `duration`
position on a yield path loses `duration ×` the largest yield increase in
the window, with the yield change in decimal (`percentage points / 100`).
An inapplicable unit/mapping pair contributes nothing. It is not recorded
as a zero loss.

## Catalog

Required historical episodes: `crash_1987`, `dotcom_2000_02`, `gfc_2008`,
`flash_2010`, `chf_2015`, `etf_2015`, `volmageddon_2018`, `covid_2020`,
`rates_2022`. `dfast_2025_severely_adverse` is a Board of Governors
hypothetical. It is not history and it is excluded from the historical
equity Mahalanobis cloud.

Primary scalars stored in code, each with a citation:

| Episode | Primary shock | Source |
|---|---|---|
| 1987-10-19 | DJIA simple return −22.6% | Brady Commission; Federal Reserve History |
| 2000–2002 | none | No peak-to-trough equity percent was verified for this build |
| GFC | S&P 500 peak-to-trough −57% | Federal Reserve History, “The Great Recession” |
| 2010-05-06 | broad-market intraday low, severe end of the stated 9–10% range | SEC/CFTC staff report |
| 2015-01-15 CHF | EURCHF day-low 0.84 vs prior day-low 1.20, simple return −30% | Bonadio, Fischer, Sauré, SJES 2025. Not a close |
| 2015-08-24 | SPY low 7.8% below the prior close | SEC research note |
| 2018-02-05 | S&P 500 −4.2% and XIV −84% | BIS Quarterly Review, March 2018 |
| COVID 2020 | none for equity | FEDS 2021-035 states the −150 bp target cut as context only |
| 2022 | none for equity | FOMC 14 Dec 2022 states the 4.25–4.50% target range as a level |
| DFAST 2025 severely adverse | equity −50%, house prices about −33%, CRE −30% | Fed 2025 stress-test scenarios. Hypothetical |

Vendor equity, Nasdaq, VIX, Wilshire, and S&P tapes are not public-domain
Board works and are not in the repository. Where this build did not read a
primary scalar, the catalog omits it.

## Public-domain bundle

`src/quant_fund/stress/public_domain/fred_h10_h15.json` is a crisis-window
extract of FRED `fredgraph.csv` retrieved 2026-09-27. Series are H.15
(`DGS10`, `DGS2`, `DFF`) and H.10 (`DEXUSEU`, `DEXSZUS`). U.S. government
work, public domain. Missing prints coded `.` were dropped. FRED vintages
can be revised; this file is that vintage, not a living feed.

EURCHF levels are `DEXSZUS × DEXUSEU` (CHF per USD times USD per EUR). Those
are noon buying rates. On 2015-01-15 the noon cross is about 1.04, a smaller
move than the −30% intraday day-low. Do not add the two.

Kenneth French daily factors can be fetched at runtime
(`quant_fund.stress.vendor_cache`). The file is academic-use data from
`mba.tuck.dartmouth.edu` and is not committed. Percent columns are converted
to decimal.

## Scenario generators

All four are research simulators. Tests compare draws with the analytic
target. They do not claim the fitted model is the data-generating process
of a market.

| Generator | Target the tests check |
|---|---|
| Stationary block bootstrap | Politis–Romano indices via `quant_fund.metrics.inference`. Synchronous blocks keep cross-correlation. An AR(1) path keeps lag-1 autocorrelation; an i.i.d. resample of the same series does not. |
| GARCH(1,1) + t copula, and a t-copula C-vine | Unconditional variance `ω/(1−α−β)`. Innovation marginal variance 1. Kendall τ `(2/π) arcsin(ρ)`. Tail dependence `2 t_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ)))` (Demarta–McNeil 2005). The C-vine uses the Aas–Czado–Frigessi–Bakken recursion; the first pair is exactly that pair-copula. |
| Gaussian HMM | Stationary distribution from `πP = π`. Unconditional mean and covariance of the Gaussian mixture. Draws start from `π`. A univariate fit can wrap `fit_markov_switching_mean`. |
| Merton jump-diffusion | Reuses `merton_jump_simulate` and `merton_log_moments`. Multi-asset Brownian shocks are correlated; jumps are idiosyncratic, so jump variance sits on the covariance diagonal only. Calibration matches sample mean and variance with fixed jump mean and jump volatility. It does not match skewness. |

Heavy Monte Carlo checks are marked `slow`.

## Reverse stress

The plausible set is `(x − μ)ᵀ Σ⁻¹ (x − μ) ≤ c²`. The default radius is the
95% Gaussian ellipsoid, `c = √(χ²_k, 0.95)`, so the boundary has
plausibility 0.05.

For linear loss `L(x) = −wᵀx` the worst point is analytic:

`x* = μ − c Σ w / √(wᵀ Σ w)`.

A general loss is searched on the Mahalanobis sphere and refined with
SLSQP. The reported plausibility score is the chi-square tail
`P(χ²_k ≥ d²)` under the supplied covariance (1 at the center). A Gaussian
density ratio `exp(−½ d²)` is reported beside it. Neither number is a
market probability.

The historical-equity cloud uses one primary `us_equity` simple-return
scalar per historical episode (five points). DFAST’s −50% is not in that
cloud. With a return panel, the covariance is the sample covariance of the
panel. A ridge is applied only when the smallest eigenvalue is at or below
`1e-10`, and the ridge size is written on the report.

## VaR and expected shortfall

`level` is the VaR confidence (0.95 means a 95% VaR). Losses are minus
portfolio returns.

Point estimates: historical and Gaussian (existing `quant_fund.metrics.risk`),
Student-t and Cornish–Fisher (existing `parametric_var_es`), and a
variance-targeted GARCH filter that scales a historical residual VaR/ES by
the last conditional sigma.

Intervals are stationary-bootstrap percentiles of the historical estimators.

Backtests on an expanding window call the existing battery:

- Kupiec and Christoffersen receive `level` (`quant_fund.metrics.var_backtest`). Expected hit rate is `1 − level`.
- Acerbi–Székely Z1 receives the tail probability `1 − level`.
- Acerbi–Székely Z2 bootstrap is `acerbi_szekely_test`.

If a test is unidentified (no hits, no transition in one state, too few
exceedances), the report stores `status: undefined` and the reason. It does
not invent a passing p-value.

## CLI

```bash
dipcatcher stress crises
dipcatcher stress report \
  --strategy configs/stress_research.yaml \
  --out stress-report.md \
  --format markdown
```

`--format html` writes a self-contained page. Text is escaped. A JSON sidecar
is written next to the report. `--returns` points at a decimal-return CSV
(optional `date` column). At least ten finite rows.

`make stress-smoke` runs the fast unit tests and this catalog report.

## Limitations

- Dot-com, COVID, and 2022 have no bundled equity-index path and no
  peak-to-trough equity scalar. Those omissions are deliberate.
- The 1987 equity shock is the Dow, not a total-return S&P path.
- The Flash Crash and the 24 August 2015 figures are intraday. Daily H.15
  yields do not replay them.
- The CHF −30% shock is a day-low versus the prior day-low. The H.10 path is
  a noon buying rate.
- XIV’s −84% applies only to an `inverse_vol_etp` weight.
- DFAST 2025 is a supervisory hypothetical, not a replay of 2008 and not the
  2024 scenario’s −55% equity decline.
- GARCH parameters come from a coarse variance-targeted grid, not a claimed
  MLE. The t degrees of freedom in the report are a kurtosis moment, not a
  claimed MLE.
- Reverse-stress plausibility is Gaussian in the covariance you handed it.
- The engine is not a trading system and not a capital model.
