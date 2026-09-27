# Agent-based limit-order-book simulator

This is a research simulator. Orders never leave the process. There is no
broker, no order router, and no live trading path. Every report is a
**simulation diagnostic** on a **SYNTHETIC** tape (`research_only`,
`live_pnl_claim=false`, `evidence_class=SIMULATION_DIAGNOSTIC`). It is not
a research-catalog score and it is not evidence about a traded book.

The package is `quant_fund.market_sim`. Reproduce a full measurement with:

```bash
uv run python -m quant_fund.market_sim report --out /tmp/market_sim_report.json
```

Smoke tests, excluding the long fact run:

```bash
make market-sim-test
```

## What is new

`quant_fund.microstructure` builds a bar-level synthetic book for feature
benches. `quant_fund.execution` prices schedules and a paper broker.
`quant_fund.backtest` fills target weights on bars. This package is a
different layer: one instrument, continuous price-time matching, and a
population of agents whose orders are the only flow.

## Matching

The core is C (`src/quant_fund/market_sim/_lob_core.c`), compiled on first
import with `cc -O3 -std=c11 -fPIC -shared -Wall -Wextra -Werror` and cached
outside the repo. `LOB_CORE_CC` overrides the compiler.

- Limit prices are integer ticks in `[1, 100000)`.
- Time priority at a price is insertion order. A nanosecond timestamp is
  stored and copied onto prints. It does not reorder the queue.
- Limit orders match through anything they cross, then rest.
- Market orders never rest. Immediate-or-cancel cancels the residual.
  Fill-or-kill either trades the full quantity or changes nothing.
- Cancels remove a live id.
- A halt stops continuous matching. Limits rest, including through the
  opposite side. Immediate orders are rejected. Market orders rest as
  market-on-open.
- The reopen is a single-price call auction. The clearing price maximises
  matched volume. Ties break toward the reference price, then toward the
  lower price. Prints use the clearing price, with the buyer as taker and
  the seller as maker.
- Trades are zero-sum in integer tick-dollars. The sum of positions is 0.
  The sum of cash is the strategy's starting cash, or 0 when no strategy
  is attached.

## Agents

Parameters live in `EcologyConfig` and were chosen before the measurement
below. They are literature-scale (Avellaneda–Stoikov risk aversion and
arrival slope, a noise-dominated population, heterogeneous momentum
horizons). They are not a fit to a sample of market data.

- A latent fundamental is a two-regime random walk. Market makers do not
  see it.
- Market makers quote the Avellaneda–Stoikov reservation price and spread
  with a one-step horizon, skip a side that would break their inventory
  cap, and cancel their previous quotes each wake.
- Momentum and mean-reversion agents trade the public mid only.
- Noise traders mix limits, markets, and cancels, with a persistent sign
  and a lognormal size.
- Informed traders see a noisy copy of the fundamental.
- One execution agent slices a parent order into market children.
- After a large trade, a few noise traders wake once. Those wakes are not
  rescheduled.

`validation_config` lengthens the same population for the fact tests.
`drought_config` is the liquidity-drought population (one wider market
maker, noise quotes further from the touch, more cancels).

## Stylized-fact rules

These rules were fixed before the tape was scored. A short sample is
**inconclusive**, not a pass. Asymptotic p-values assume independent
observations and are anti-conservative under dependence.

| Fact | Test | Pass |
|---|---|---|
| Fat tails | Jarque–Bera | p < 0.05 and excess kurtosis > 1, n ≥ 200 |
| Volatility clustering | Engle ARCH-LM, 5 lags | p < 0.05, n ≥ 200 |
| No return autocorrelation | Ljung–Box, lag 10, on signed returns | do **not** reject at 5% |
| Long memory of absolute returns | DFA Hurst and Ljung–Box on \|r\| | H(\|r\|) ≥ 0.60, H(r) in [0.35, 0.65], Ljung–Box on \|r\| rejects, n ≥ 256 |
| Square-root impact | log-log OLS of adverse ticks on filled/volume | 95% interval contains 0.5 and excludes 0, at least 12 positive points |
| Spread and depth shape | D'Agostino skewness test and AIC | both series: positive skew at 5% and lognormal (location fixed at 0) beats a normal, n ≥ 100 each |

Returns are non-overlapping log mid changes every `return_stride` trades
after warmup. Spread and depth are sampled every `sample_every` events
when the book is two-sided and not halted.

The impact experiment is separate from the ecology tape. A buy metaorder
of size 50, 100, 200, 400, 800, or 1600 shares is sliced into 20 children,
one every 8 events, starting at event 600. Three seeds are used (7, 8, 9).
Arrival is the mid immediately before the first child. Adverse impact is
the volume-weighted average price minus that mid, in ticks. Volume counts
each print once. Non-positive adverse impact is dropped and counted. If
the interval contains 0.5 and also contains 0, the result is inconclusive.
The slope is not adjusted after the run.

## Scenarios

Each scenario is the same seeded ecology plus a hook, except the drought,
which replaces the population.

- **Flash crash.** Market-maker quotes are cancelled, makers are paused,
  and a market sell larger than the visible bid walks the book. Makers
  resume later.
- **Halt and auction.** Continuous matching stops, crossed interest is
  resting, and a call auction reopens the book. Trades during the halt
  must be zero.
- **Liquidity drought.** The drought population above. Compared with the
  baseline by median spread, not by a one-shot order.
- **Gap open.** Resting orders are cancelled, the latent price jumps, a
  new ladder is posted, and a call auction reopens.
- **Crowded unwind.** A buy burst lifts the book, then a larger sell
  burst hits it.

## Strategy stress

A strategy is a function from the closes so far to one target weight.
`lightspeed_momentum_weight` calls `quant_fund.lightspeed.momentum.momentum_scores`
on the single name `SIM` and scales the last score by 4. That scale is an
adapter from a score to a weight, not a fit. `mean_reversion_weight` fades
a 20-bar trailing mean. `target_weight_replay` walks a
`(event_time, security_id, target_weight)` panel in order.

The harness also runs `run_backtest` on the **background** tape (the same
seed with no strategy orders), next-open fills, `configs/research.yaml`.
The in-memory risk limits are widened so the weight is executable:
`max_name` 1, `max_net` 1, `max_gross` 2, `max_participation` 1,
`max_order_notional` 1e12, `participation_limit` 1, `stale_price_bars`
10000, `max_predicted_vol` 5. The data root is an empty temporary
directory, so a host GARCH file cannot enter the gate. The research config
on disk is not modified.

The strategy's orders change the book, so its tape and the background tape
differ. `weight_l1` is the mean absolute gap between the weights on those
two close paths. Mark-to-market change and drawdown are path diagnostics.
Slippage is the volume-weighted adverse gap versus the decision mid, in
basis points of that mid. The plain backtest charges the research cost
model; the matching engine does not charge commission. The two numbers
are not two measurements of one fill.

Dictionary keys do not use Sharpe, Sortino, Calmar, P&L, or NAV tokens.
Those quantities are not research headlines here.

## Performance method

`matching_benchmark` runs the C core only.

1. Pass 1 draws a tape with xorshift64 and matches it: 15% market, 15%
   aggressive limit, 30% cancel at the touch, 40% passive limit. The
   opening ladder is 30 levels of quantity 40 around tick 50000.
2. The order arena of a fresh book is prefaulted with `memset`, then the
   same ladder is posted.
3. `CLOCK_MONOTONIC` wraps pass 2 only: `lob_submit` of the recorded
   events. No Python, no trade log.
4. Pass 2 must reproduce pass 1's checksum and trade count.

`match_events_per_s` is events divided by the pass-2 elapsed time.
`generate_and_match_events_per_s` is pass 1, which includes generation.
The gate is one million **match** events per second, single-threaded.
The figure below is one run on the authoring machine. It is not a
best-of-n sample.

## Measured run

The tables in this section are copied from one `report` invocation. If a
fact failed or was inconclusive, that is the result.

One invocation of `python -m quant_fund.market_sim report --events 1000000`
on the authoring machine. The raw JSON is the measurement; the tables
below are that file, not a second run.

Machine: 4 vCPU Intel Xeon (KVM guest), compiler `Ubuntu clang version
18.1.3`, flags `cc -O3 -std=c11 -fPIC -shared -Wall -Wextra -Werror`.
Seed 1 for the matching benchmark. Ecology seed 7 unless a table says
otherwise. No parameter was changed after these numbers were produced.

### Matching throughput

| Quantity | Value |
|---|---|
| Events | 1,000,000 |
| Pass-2 elapsed | 29.643421 ms (`elapsed_match_ns` = 29,643,421) |
| Match throughput | 33,734,298 events/s |
| Pass-1 elapsed (generate and match) | 50.728270 ms |
| Generate-and-match throughput | 19,712,874 events/s |
| Trades | 379,152 |
| Rejects | 0 |
| `list_ok` | true |
| Return code | 0 |

The pass-2 checksum matched pass 1 (checksum `-3184260662416822417`).
The match figure is above the one-million-events-per-second bar. It is
one timed replay after the arena prefault, not a minimum over repeats.

### Stylized facts

`validation_config(seed=7)`: 24,000 events, 4,765 trades, 1,066 returns,
7,334 spread and depth samples, 960 bars. Positions summed to 0 and cash
summed to 0.

| Fact | Status | What the test returned |
|---|---|---|
| Fat tails | pass | Jarque–Bera stat 6,115.98, p reported as 0, excess kurtosis 11.67, skew -0.61 |
| Volatility clustering | pass | ARCH-LM stat 99.46, p 6.86e-20 |
| No return autocorrelation | fail | Ljung–Box stat 57.62, p 1.02e-8, \|rho1\| 0.116. Signed returns are autocorrelated. |
| Long memory of absolute returns | fail | H(\|r\|) 0.603, H(r) 0.682, Ljung–Box on \|r\| p 6.35e-8. Absolute returns are persistent, and so are signed returns, so the joint rule fails. |
| Square-root impact | inconclusive | 18 of 18 trials had positive adverse impact (0 dropped). Slope 2.14, 95% interval [-1.42, 5.70], R² 0.092. The interval contains 0.5 and also contains 0. |
| Spread and depth shape | fail | Spread n=7,334, skew -0.48, skewness p about 0, normal AIC 9,297 beats lognormal AIC 10,290. Depth n=7,334, skew 0.023, skewness p 0.43, normal AIC 84,398 beats lognormal AIC 84,635. |

The impact design was the pre-registered buy schedule (sizes 50 through
1,600, three seeds, 20 slices). The slope was not adjusted.

### Scenarios

Default `EcologyConfig` (seed 7, 2,500 events), except the spread
comparison, which uses that config and `drought_config` of it.

| Scenario | Measured path |
|---|---|
| Flash crash | 9 market-maker orders pulled, market sell filled 283, last print 11 ticks under the prior mid (19,999), later recovery fraction 0.50. 662 trades. |
| Halt and auction | 0 trades while halted. Call auction matched 166 shares at tick 19,998. 588 trades in the whole run. |
| Gap open | Prior mid 20,000.5. Auction price 20,399, so the reopen cleared 398.5 ticks above the old mid (the latent jump was 400). Auction quantity 346. 650 trades. |
| Crowded unwind | Buy burst filled 369, mid moved from 19,999 to a peak of 20,006.5. Sell burst filled 227, last print 19,988. 807 trades. |
| Liquidity drought | Population change, no order hook. Median spread 58 ticks (531 samples) against the baseline median of 3 ticks (563 samples). 1,040 trades inside the drought run. |

### Strategy stress against the plain backtest

Same default ecology. Weights are clipped to `strategy_max_weight` (0.5)
on both paths. Mark-to-market change is the last marked value divided by
the first, minus one. Drawdown is the minimum of that path's simple
returns. Slippage is basis points of the decision mid. The plain backtest
is next-open fills on the background tape with the diagnostic risk limits
described above.

Quiet scenarios barely move the 20-bar mean or the momentum score, so the
weight stays near zero and both paths trade a small number of shares.
The gap, the drought, and (for mean reversion) the crowded unwind push
the weight to the cap. Share targets are `weight * marked_value / mid`.
When the mid leaves the opening region, that target becomes a very large
share count, most of it does not fill, and the slippage number is no
longer a small-order statistic. Those rows are reported as measured.

Mean reversion:

| Scenario | Sim mtm change | Sim max drawdown | Slippage bps | Filled / requested | Backtest mtm change | Backtest max drawdown | Risk-gate rejects | Weight L1 |
|---|---|---|---|---|---|---|---|---|
| flash_crash | -2.58e-6 | -2.58e-6 | 0.94 | 134 / 151 | -2.24e-5 | -2.26e-5 | 0 | 0.00077 |
| halt_auction | -1.68e-6 | -1.68e-6 | 0.72 | 111 / 111 | -9.50e-6 | -9.50e-6 | 0 | 0.000035 |
| liquidity_drought | 0.177 | -0.0615 | 4,422 | 806 / 68,361,743 | -3.40e-4 | -3.90e-4 | 1 | 0.197 |
| gap_open | -0.443 | -0.561 | 10,098 | 1,726 / 285,847,097 | -3.69e-3 | -5.37e-3 | 2 | 0.222 |
| crowded_unwind | 0.0108 | -0.164 | 3,542 | 1,221 / 221,954,781 | -2.96e-5 | -2.97e-5 | 0 | 0.213 |

Lightspeed momentum score, scaled by 4:

| Scenario | Sim mtm change | Sim max drawdown | Slippage bps | Filled / requested | Backtest mtm change | Backtest max drawdown | Risk-gate rejects | Weight L1 |
|---|---|---|---|---|---|---|---|---|
| flash_crash | -3.08e-6 | -3.08e-6 | 1.07 | 126 / 139 | -2.18e-5 | -2.30e-5 | 0 | 0.00052 |
| halt_auction | -1.18e-6 | -1.21e-6 | 0.58 | 100 / 100 | -6.98e-6 | -6.98e-6 | 0 | 0.00029 |
| liquidity_drought | 0.0654 | -0.103 | 4,897 | 727 / 69,702,555 | -2.15e-4 | -2.84e-4 | 0 | 0.225 |
| gap_open | -0.335 | -0.461 | 4,527 | 1,626 / 113,111,706 | -4.44e-3 | -4.44e-3 | 0 | 0.262 |
| crowded_unwind | -6.11e-6 | -6.11e-6 | 1.04 | 224 / 264 | -2.76e-5 | -2.76e-5 | 0 | 0.00115 |

`weight_l1` near zero means the strategy tape and the background tape
produced almost the same weights. On the gap and the drought the tapes
diverge, and the in-book result is not the plain backtest. Both columns
are simulation diagnostics. Neither is a live result.

## Limitations

- One instrument, integer ticks, no fees inside the matching engine, no
  latency, no partial time priority beyond insertion order.
- Agent parameters were not estimated from a market sample. A passing fact
  test says the synthetic tape met a statistical rule. It does not say the
  tape is a market.
- p-values ignore dependence.
- The square-root law is a property of this sliced-buy design. A fail or
  an inconclusive fit means this design did not recover the law.
- Strategy stress and the plain backtest see different prices, and only
  the backtest charges the cost model.
- Nothing here is a forecast, a capacity estimate, or a live result.
