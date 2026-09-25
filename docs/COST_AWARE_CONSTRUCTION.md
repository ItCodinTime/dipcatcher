# Cost-aware portfolio construction (step 3)

The net-return tournament now supports an optional convex allocator. It uses
actual filled positions, forecast and uncertainty proxies, covariance and
modeled costs to choose target weights. It can keep cash or leave positions
unchanged when the expected benefit of trading is too small.

This is an engineering implementation, not measured market uplift. The tracked
real-price Parquet snapshot can be recovered from a normal Git checkout.
Generated-price tests verify mechanics; they do not establish profitability.
The existing vendor universe has survivorship bias and an already-inspected
holdout. A rejected CLARABEL solve records its status, iteration count and
solve time when available, with `weights_accepted=false`; it does not fall
back to an approximate portfolio.

The 2026-09-25 frozen snapshot run is a blocked validation: both cost-aware
variants returned `optimal_inaccurate`. Its sealed failure and control ledgers
are in `data/metadata/research/phase1_20260925.md`, linked by
`data/metadata/research/phase1_evidence_index.json`. There is no selected
candidate and no test receipt. The test command below applies only after a
new frozen run has selected a validation candidate.

## Run a matched comparison

After preparing the benchmark manifest as described in REAL_DATA_BENCHMARK.md:

```bash
PYTHONPATH=src python -m quant_fund.research.net_tournament prepare \
  --benchmark-run data/metadata/real_benchmark \
  --spec configs/cost_aware_tournament.json \
  --output data/metadata/cost_aware_tournament
PYTHONPATH=src python -m quant_fund.research.net_tournament run \
  --run data/metadata/cost_aware_tournament --phase validation
PYTHONPATH=src python -m quant_fund.research.net_tournament run \
  --run data/metadata/cost_aware_tournament --phase test
```

The sample slate freezes momentum-20 and reversal-1, each with a rank control
and cost-aware variant, plus the equal-weight benchmark. A cost-aware candidate
must have an otherwise identical rank control. The default configuration is an
illustrative protocol, not a fitted optimum. Every parameter change is a new
research trial and requires a new frozen run; record external experiments too.

All strategies use the same universe and history requirement (the largest
lookback/risk window across the slate), sessions, next-open execution, costs,
initial capital and exposure limits. Adding a longer risk window can change the
common universe, so compare within the same frozen run. The optimized allocator
can change exposure, selection persistence and turnover; this is a construction
ablation, not a risk-matched or factor-neutral alpha estimate.

Each scenario includes `allocation_ablations`: descriptive differences in mean
daily net return, total return, maximum drawdown (negative convention), aggregate
costs and traded notional. Differences are candidate minus control. Pairwise
comparisons do not claim statistical significance. The existing full-slate
RC/SPA/StepM procedure still compares every candidate with equal weight;
validation-only selection and failure/terminal-liquidation gates still apply.
A failed allocator remains a failed candidate and blocks selection. Nothing in
this runner promotes a strategy or enables live trading.

## Model and units

Weights are fractions of signal-close NAV. Previous weights come from filled
shares marked at that close, including drift and prior fees, not yesterday's
target. With `d = |w - previous|`, the allocator maximizes:

```
alpha @ w - risk_aversion * w.T @ covariance @ w
          - uncertainty_aversion * uncertainty @ abs(w)
          - linear_cost * sum(d) - impact @ d**1.5 - holding_cost(w)
```

| Input | Construction / units |
|---|---|
| Return proxy | Daily geometric rate of the same trailing price ratio used to rank names, multiplied by frozen `alpha_scale`; negated for reversal. Unselected existing positions get zero alpha. This is not a trained return forecast. |
| Covariance | Sample covariance of simple close returns over `risk_window`, shrunk toward its diagonal with a frozen coefficient. No future bars or labels. |
| Uncertainty | `sqrt(diag(covariance) / lookback)`. A dispersion proxy with an independence approximation, not a calibrated confidence interval. |
| Linear costs | `(commission_bps + half_spread_bps) / 10000` per traded NAV fraction. |
| Impact coefficient | `impact_y * known_20_session_volatility * sqrt(NAV / known_ADV_dollars)`. Multiplied by `d**1.5`, this matches the replay's square-root participation impact at planning prices. |
| Holding costs | Short borrow plus financing, with cash return expressed as opportunity cost. APR / 252 is the fixed one-session planning approximation; replay charges actual calendar time on its cash/positions ledger. Funding APR must be at least cash APR. |

Forecast magnitudes and uncertainty are explicit proxies. Before making an
investment claim they need train-only calibration, separate validation and
independent forward evidence. The public `allocate` function also accepts
externally generated as-of forecasts and uncertainty arrays, but timestamp and
training provenance for those inputs remain the caller's responsibility.

## Constraints and execution

- Total absolute change is capped by `turnover_limit`; each change is capped by
  `participation_limit * known_ADV / signal_NAV`.
- Buffered name/gross limits include estimated transaction fees in their NAV
  denominator. Unlevered portfolios retain the declared cash buffer after
  estimated trading fees. Holding fees and future price gaps are approximated,
  not guaranteed by the planning constraints.
- New exposure is allowed only in the names and directions selected by the
  control's rank signal. Existing unselected positions may stay or shrink,
  allowing gradual exits when capacity or costs bind. There is no mandatory
  full investment or net-neutrality constraint.
- Existing holdings need available marks, risk history and liquidity. Missing
  inputs fail the candidate. If drift, liquidity or turnover constraints make
  the problem infeasible, it fails without relaxing limits or substituting
  target/previous weights.
- CLARABEL must return `optimal`. Returned weights are checked against every
  constraint with a maximum absolute residual of `1e-7` NAV fraction. Tiny
  changes below `1e-8` are snapped to previous weights and rechecked. These are
  numerical tolerances, not an economic minimum-order policy.
- Next-open prices can differ from planning prices. Replay still caps fills and
  checks realized cash/exposures, recording rejects and unfilled amounts. The
  next solve uses actual filled holdings. Terminal exits bypass the optimizer's
  turnover budget but still obey execution participation and risk/cash checks;
  residual positions are marked and prevent the evidence gate from passing.
- The double-impact scenario increases realized execution costs; planning
  coefficients stay frozen. Later decisions can differ because costs changed
  cash/holdings. This stresses the same policy under worse implementation costs.

The `allocations` ledger records security mapping, planning NAV, prior/target
weights, alpha/uncertainty/capacity, predicted risk and costs, turnover, solver
status and constraint residual. Tournament receipts hash allocator code and
record CVXPY, CLARABEL and SciPy versions in addition to the prior runtime stamp.
Old receipts require their original code/runtime; create a new run for this
implementation.

## Validation and research basis

Focused tests cover analytical risk optima, cost-induced no-trade regions,
impact/uncertainty response, borrow/cash costs, liquidity/turnover/post-fee
exposures, invalid inputs, solver/infeasibility failures, causal decisions,
actual-fill feedback, matched ablations and frozen validation/test execution.
A flat generated market demonstrates avoided round-trip costs; it is a
controlled accounting example rather than evidence of a market edge.

The objective follows the single-period convex construction framework in
[Boyd et al., Multi-Period Trading via Convex Optimization (2017)](https://stanford.edu/~boyd/papers/cvx_portfolio.html),
including linear and 3/2-power transaction costs and holding costs. See the
[authors' slides](https://stanford.edu/~boyd/papers/pdf/cvx_portfolio_talk.pdf)
for the normalized impact model. This implementation is single-period and
does not implement the paper's multi-period forecasting/control system.
The legacy `portfolio.optimizer` API is unchanged; this bounded research
allocator is integrated specifically with step 2's independently audited replay.
