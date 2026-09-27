# Backtest-to-live parity

How to read a parity report. The report compares two **simulated** replays of
one strategy: a backtest-timed reference and a shadow replay. Both call the
same `decide` function and both send orders only to `SimulatedBroker`.

Nothing in this report is a live fill, a broker acknowledgement, or a
research score. `live_pnl_claim` is false. `would_promote_live` is false.
A SYNTHETIC tape is labeled `data_source: SYNTHETIC` and is not market
evidence.

```bash
uv run python -m quant_fund.parity smoke --out data/metadata/parity-smoke
```

That smoke writes the same files a real comparison writes, on a four-day
SYNTHETIC tape, and exits non-zero unless every bar matches.

## Files

| File | What it is |
|---|---|
| `summary.json` | Verdict: match flag, divergence rate, first divergence, terminal gap |
| `divergence_ledger.json` | One row per divergent `(bar, security)`, with one cause |
| `shortfall.json` | Paper-versus-backtest marked-value gap, split into six components |
| `code_path.json` | Import graph, static callees, runtime call trace |
| `report.md` | The same numbers in reading order |

`summary.match` is true only when the decision ledgers match **and** the
code-path guard agrees. `decisions_match` can be true while
`same_code_path` is false: the weights agreed, but one origin called a
function the other did not.

## Summary statistics

| Field | Meaning |
|---|---|
| `n_bars` | Union of `(event_time, security_id)` keys |
| `n_matched` | Keys present on both sides with no differing field |
| `n_divergent` | Keys that differ, including a bar present on only one side |
| `divergence_rate` | `n_divergent / n_bars` |
| `by_cause` | Count of divergent rows for each cause below |
| `first_divergence` | Earliest divergent bar in time, then security id |
| `max_abs_weight_delta` | Largest absolute rounded-weight gap, missing side treated as zero |
| `terminal_gap` | Backtest marked-value change minus paper marked-value change |

Marked-value change is `final_marked_value - initial_cash` inside the
simulator. It is the gap the shortfall section decomposes. It is not a
headline P&L, not NAV, and not a Sharpe ratio.

A divergence rate of zero means these two runs agreed. It does not mean
the strategy made money, and it does not mean a later live session would
agree. Live order routing is out of scope and is refused if
`runtime.mode` is `live`.

## Divergence ledger

Rows line up on the bar's `event_time` and `security_id`. Each divergent
row has exactly one `cause`. Precedence is fixed, top to bottom, so a
later symptom is not labeled as the root:

1. **`data`** — source, revision, release time (`available_time`), or the
   decision-bar / execution-bar tape differs. A bar present on only one
   tape is `missing_bar`. A revision that changes the close but not the
   next open is still `data`, even when the fill price matches and the
   terminal gap is about zero.
2. **`timing`** — `decision_time` differs. That is the bar-close clock
   versus a lagged or early decision. A row whose `decision_time` lines
   up with the other side's row at a different `event_time` is
   `shifted_bar`.
3. **`code_path`** — the call-trace digest differs (`call_trace`). If
   every input matches and the target weight still differs, the detail
   is `output_mismatch`: the strategy returned something else from the
   same observed inputs.
4. **`state_drift`** — broker cash, positions, or strategy state differ.
   After a restart the detail is `after_restart`, including later bars
   whose state never reconverged. A state mismatch with no restart and
   no earlier cause is `state_digest`. A cost-parameter change that also
   moves cash is **`costs`**, not state drift: restart generation is
   what promotes a state mismatch ahead of costs and fills.
5. **`rounding`** — `lot_size` differs, or the raw weights match and the
   lot-rounded weights do not.
6. **`costs`** — `commission_bps`, `half_spread_bps`, `impact_y`, or
   `bps_per_turnover` differ.
7. **`fills`** — fill quantity, fill price, or explicit cost dollars
   differ after the tape, clock, trace, restart state, lot, and cost
   parameters match.

The ledger stores the backtest and shadow revision, rounded weight, fill
price, and fill quantity beside the cause so you can see the symptom the
precedence chose not to name.

## Implementation shortfall

`terminal_gap` is backtest marked-value change minus paper marked-value
change. Positive means the paper book underperformed the backtest book
inside the simulator.

| Component | What it measures |
|---|---|
| `delay` | Shared signed quantity times (paper exec price − backtest exec price) |
| `spread` | Paper spread dollars minus backtest spread dollars |
| `impact` | Paper impact dollars minus backtest impact dollars |
| `fees` | Remaining explicit cost (commission, plus turnover bps when the broker charged it), paper minus backtest |
| `missed_fills` | Backtest-only quantity, marked from the backtest exec price to the backtest terminal print |
| `opportunity` | Paper-only quantity, signed so it explains the gap, plus any shared-quantity gap in the terminal print |

Fills pair in listed order within `(event_time, security_id)`. The six
numbers sum to `fill_gap`. `algebraic_residual` is that identity and
should be numerical dust. `residual` compares the same sum to the books'
terminal gap; dust means the books contain no cash flow the fill tape
does not explain.

`opportunity_detail.shared_mark_gap` is the part of opportunity that
comes from marking the same position at two different terminal prints
(a vendor or revision difference in the last close). It is not delay.

`paper_arrival_shortfall` is the existing per-fill implementation
shortfall against the **decision** price (`execution.implementation_shortfall`).
`in_terminal_gap` is false. Do not add it to the six components.

A data-revision divergence can have a terminal gap of zero when prices
that affect fills and the terminal print did not change. Read the ledger
and the shortfall together.

## Code-path guard

Three comparisons, all of which must hold for `same_code_path`:

| Check | What "equal" means |
|---|---|
| Import graph | `quant_fund.parity.reference` and `quant_fund.parity.shadow` reach the same imports, and the two `decide` modules do too |
| Static callees | The AST call names inside each `decide` function match |
| Runtime trace | The ordered `module:qualname` list recorded while `decide` ran matches |

Runtime recording uses `sys.setprofile` and does not replace a coverage
line tracer. A branch on `ctx.origin` that calls a helper on only one
side shows up in `runtime_only_backtest` or `runtime_only_shadow` even
when the returned weight is identical. That is a code-path divergence
with a zero terminal gap: the weights hid the branch, the trace did not.

`same_code_path` does not prove the strategy is free of lookahead. It
proves these two runs did not take observably different functions.

## What the replay actually does

* Recorded sessions (`MarketSession`) can be replayed twice. A streamed
  iterable is consumed once, one bar group at a time, with no lookahead
  past the group being processed.
* `pacing="accelerated"` does not sleep. `pacing="wall_clock"` sleeps
  `(decision_time - previous) / speed` through the injected sleeper
  (default `time.sleep`). Speed `86400` turns a one-day bar step into
  one second.
* Decisions at the bar close see a bar only when
  `available_time <= decision_time`. A negative `decision_lag` is an
  early clock and will not see that bar's close.
* Fills follow the config. The default is next open (ADR-005). The last
  next-open decision is recorded and left unfilled. A close auction
  fills on the decision bar's close when `allow_close_auction` is set.
* Orders are built with `SimulatedBroker.target_to_orders` and submitted
  with `SimulatedBroker.submit`. The parity package does not construct
  any other broker.
* A restart writes `SimulatedBroker.to_dict`, optionally mutates that
  dict, and restores with `SimulatedBroker.from_state`. A clean restart
  is not a divergence. A mutated book is `state_drift`.

## Limitations

* This is not the event-driven backtest engine's internal book. It is a
  second replay of the **strategy** through the simulated broker under
  the same fill convention. Differences between `run_backtest` and
  `SimulatedBroker` are out of scope of this ledger; compare those books
  only after both have been reduced to the decision rows this package
  emits.
* One cause per bar. A restart that later changes a fill is still
  `state_drift` for every bar whose state did not reconverge.
* Participation caps and risk-gate rejects show up as `fills` (quantity
  zero or smaller) unless an earlier cause already fired.
* A held name with no print is not flattened at a carried mark. If the
  hole lasts longer than `risk_gate.stale_price_bars`, the replay raises
  `ParityValuationError` instead of inventing a price.
* No vendor feed, no drop copy, no real clock synchronization beyond the
  paced sleeper, and no live order path.
