# Forward shadow record: prospective power and freeze plan

Status: **pre-collection power analysis, 2026-09-25**. No prospective decisions or outcomes have been collected under this plan. Planning inputs are bound to the sealed `data/metadata/research/phase1_evidence_index.json` receipt SHA-256 `0ce794b56249952fce5b2ff1046eea9e50b2f4e6d691539b8019959131873204` and its net-tournament validation receipt SHA-256 `c2be98ce3e855af08819d7273b7df6b5c087a7b5107b57535ad8437a07d692c7`. Commit and externally timestamp the exact strategy protocol **before** the first new decision. The 2025 outcome and local preliminary `us_wide_v1`/`us_wide_v2` runs have already been inspected and cannot be reused as new forward evidence. The recovered price file extends to 2026-09-18, but every day in it predates this protocol freeze.

## Question and candidate

The proposed primary comparison is the frozen momentum-20 strategy against the frozen equal-weight comparator, using the tournament's common point-in-time universe and identical capital, risk bounds, next-open simulated fills, costs, borrow, financing, and terminal accounting. Momentum-20 was selected by the existing validation protocol; its sealed validation mean daily net difference was **−0.720 basis points**, so there is currently no positive edge estimate. The inspected test set also disfavored it. This plan is a falsifiable prospective check of whether a materially positive net differential emerges, not a forecast that it will.

Before starting, freeze the exact code commit and dirty-worktree state (ideally clean), configuration and protocol hashes, vendor data/source permissions, strategy and comparator, market calendar and eligibility rule, modeled costs, timestamp convention, failure accounting, test direction, minimum effect, and sample size in an externally timestamped record. Do not tune the strategy, omit failed days, or substitute a different benchmark during collection. If the existing `dipcatcher paper` command cannot express this exact strategy/comparator and ledger contract, implement and test that simulated-only adapter first; no broker connectivity is implied here.

## Effect, dependence, and sample-size calculation

Let `D_t = R_t(momentum_20) − R_t(equal_weight)`, where each `R_t` is the modeled daily **net** return from the same forward session. Predeclare one primary one-sided null `H0: E[D_t] ≤ 0` against `H1: E[D_t] > 0`; define a **material planning effect** of +5 bps per eligible session. The inferential threshold is one-sided `α=0.05`, target power `1−β=0.80`, and one fixed analysis after the required sample is reached. These are planning assumptions, not evidence that +5 bps/day is attainable. If more candidates or primary outcomes are tested, recompute the multiplicity-adjusted sample size before starting.

The sealed 495-session validation differential has an ordinary sample standard deviation of 75.222 bps. A Bartlett/Newey–West long-run standard deviation using 10 lags was 63.777 bps; lags 5, 20, and 30 gave 63.415, 65.208, and 65.018 bps. A 1,000-resample stationary bootstrap with average block length 10 (seed 20260925) gave a 95th-percentile long-run standard deviation of 74.976 bps. The realized serial-correlation adjustment reduced variance here, so planning conservatively uses the **maximum** of those estimates and the doubled-impact ordinary standard deviation: `σ* = 75.224 bps` rounded upward. This avoids assuming that negative autocorrelation will persist. The doubled-impact series had ordinary SD 75.223 bps and 10-lag long-run SD 63.780 bps.

With one prospective comparison and fixed-horizon normal approximation:

\[
N = \left\lceil
\frac{(z_{0.95}+z_{0.80})^2(\sigma^*)^2}{\delta^2}
\right\rceil,
\quad z_{0.95}=1.645,\quad z_{0.80}=0.842.
\]

| Planning effect \(\delta\) | Required eligible sessions | Approximate 252-session years |
|---|---:|---:|
| +2 bps/day | 8,747 | 34.7 |
| **+5 bps/day (primary planning target)** | **1,400** | **5.6** |
| +10 bps/day | 350 | 1.4 |

These are approximate power calculations, not a guaranteed finishing date or a guarantee of significance. They assume the prospective long-run variance resembles the validation planning bound, finite-variance behavior adequate for the planned test, no undisclosed repeated looks, and no shift in data availability or execution conditions. Five basis points per day is a large economic effect; smaller effects need a much longer record. Because the retrospective estimate was negative, a prospective positive result should be regarded as particularly surprising and subjected to replication.

The frozen prospective protocol must use at least **1,400 eligible paired sessions** if momentum-20 remains the exact candidate and +5 bps remains the planning effect. If the candidate, data source, cost convention, primary question, or variance estimate changes materially before starting, publish a new power calculation and forward protocol before collecting any outcomes under it. Never reduce the horizon by selecting a favorable interim variance.

## Collection and decision rule

Collect at least the predeclared `N` **eligible paired sessions**, measured from the first session strictly after the externally recorded freeze. A session is eligible only if both strategies could make a decision using information available at their stated cutoff and could be replayed under the frozen execution contract; track missing and ineligible sessions, failures, rejects, and downtime separately. Do not exclude a loss, reject, or feasible no-trade session merely because it worsens a score. If data availability breaks the contract, preserve the attempt and disclose the interruption; define any restart in advance as a new version rather than retrospectively shifting start dates.

For each decision persist the externally timestamped inputs and decision cutoff, intended quantities, simulated orders and fills (including partials), rejects and reasons, positions, cash, costs, NAV and net-return accounting, kill-switch state, software/config/data hashes, broker-state cursor, and immutable receipt chain. Reconcile the cursor, prior fills, cash and positions after every restart before advancing. The same ledger must produce both configured-impact and doubled-impact sensitivity views. Normal health checks may run during collection; they do not expose a primary significance result or authorize strategy changes.

At the sole predeclared analysis, report the entire paired differential stream, mean, a dependence-robust confidence interval and p-value (10-session block length plus sensitivity at 5/20/30), both impact scenarios, all failed/ineligible dates, costs, and any protocol deviations. A success claim requires a positive primary mean and one-sided adjusted p-value below 0.05, positive differential under doubled impact, completed reconciliations and terminal accounting, and intact receipts. A failure or insufficient power is reported as such. Even a passing shadow result is simulated research evidence; it does not satisfy the separate licensed-data, broker reconciliation, venue measurement, independent review, and explicit authorization gates for live trading.
