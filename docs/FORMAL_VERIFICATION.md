# Formal verification

Simulation and library checks of the order lifecycle and of the accounting
identities. This page does not describe a live broker, and it does not
report a research score.

The checked code is `SimulatedBroker`, the paper loop's order rows, the
cost function, and `adjust_prices`. New code lives in `quant_fund.formal`,
`spec/tla/`, and `tests/formal/`. Run the lane with `make formal`.

## What is new

Retail research code usually tests these paths with examples. This lane
exhausts a finite state machine and discharges symbolic identities.

- A TLA+ specification of one order (`spec/tla/OrderLifecycle.tla`),
  model-checked with TLC. Statuses are `absent`, `new`, `partial`,
  `filled`, `canceled`, `rejected`, and `expired`. The model includes
  restarts (`checkpoint` / `crash`), duplicate fill ids, late fills, and
  cancel/fill races.
- A Python trace validator with the same transition relation over positive
  quantities. It is aligned with the TLA module by hand. TLC does not
  check that alignment.
- Conformance traces from `SimulatedBroker` (via `TraceSession`, which
  records the requested quantity before submit rewrites an IOC child) and
  from `run_paper_loop` order rows.
- Z3 proofs, through the `z3` Python bindings, that the accounting closed
  forms hold for every real assignment of their preconditions. Each proof
  is `unsat` of the negated claim. The solver timeout is 20 seconds; none
  of these queries needed it.
- Hypothesis rule-based state machines that drive the simulated broker
  against `Account` and against the trace validator.

## Order lifecycle

`new` is the broker status `ACKED`: working, nothing filled. `CANCELLED`
with `reject_reason == "expired"` is `expired`. Other cancels are
`canceled`. `partial` means a residual is still working. A participation
cap that does not rest is terminal `filled` with `filled <= qty`. History
rows stamp the child slice `FILLED` even when `is_partial`; the trace uses
`is_partial` and whether a residual remains.

`amend` replaces the residual. The specification quantity is the new total
authorization, already-filled plus the new residual, and it must be strictly
above filled.

Restart of a durable snapshot is `checkpoint` then `crash`. `crash` without
a new checkpoint drops uncommitted fills. The paper ledger writes the
snapshot after the step, so an observed resume is the identity. The
specification still admits the lossy crash. `from_state` dedupes fill ids
when rebuilding `fills` and restores cash from the snapshot; it does not
re-apply fills.

Shadow orders (`allow_capital=False`) ack and do not fill. Finite traces
may stay non-terminal. Liveness is not claimed for them.

### TLC

`scripts/run_tlc.sh` downloads `tla2tools.jar` from the TLA+ 1.8.0 release
(`v1.8.0`, sha256
`ab4694601923fd5ac06452abbf847c366a5054a3d739552085edd6ed986c29ec`)
into `.tla/` (gitignored), checks the digest, and runs TLC with `-workers 2`
and `-Xmx1g`. The jar is not source.

Measured on this machine (Linux, Java `Ubuntu 21.0.10`, TLC
`2026.09.25.163503`) on 2026-09-27. Both runs printed
`Model checking completed. No error has been found.` and `Finished in 00s`.

| Config | Bound | Invariants and properties | Generated | Distinct | Depth |
|---|---|---|---|---|---|
| `OrderLifecycle.cfg` (`SafetySpec`) | `MaxQty=3`, `MaxFillId=2` | `TypeOK`, `NeverOverfilled`, `TerminalShape`, `FillAccounting`, `CheckpointPrefix`, property `ActionSafety` | 3523 | 345 | 7 |
| `OrderLifecycleLiveness.cfg` (`LivenessSpec`) | `MaxQty=2`, `MaxFillId=2` | `TypeOK`, `NeverOverfilled`, `TerminalShape`, `FillAccounting`, property `EventuallyTerminal` | 428 | 76 | 5 |

`ActionSafety` is the conjunction of: a fill step ends in `partial` or
`filled`; a durable terminal snapshot does not change; a terminal status
that matches its snapshot stays terminal; the durable fill set and durable
filled quantity are monotone.

`EventuallyTerminal` is `(status ∈ Working) ~> (status ∈ Terminal)` under
weak fairness of `Resolve` (reject, cancel, expire, or fill). `LivenessSpec`
omits `Crash` and `Amend`. An adversary that crashes forever, or that
amends the residual upward forever, can postpone termination. Those
behaviors are outside the liveness claim.

Optimistic fingerprint-collision estimates printed by TLC were `5.9E-14`
(safety) and `1.5E-15` (liveness).

The Python enumerator `enumerate_safety(max_qty=2, max_fill_id=2)` visited
99 states and reported no invariant violations. That enumerator is not
TLC, and 99 is not the TLA distinct-state count: the bounds and the
encoding of fill ids differ.

## Conformance

`tests/formal/test_order_lifecycle.py` drives the simulated broker through
a market fill, a price-missing reject, a resting partial then amend then
completion, a cancel, an expire that wins over a touching bar, and a
restart. `TraceSession` events and `events_from_broker` are both required
to be allowed traces, and they agree on status, filled quantity, and
authorization.

An IOC partial keeps the parent quantity in the session trace (requested
40_000, filled 100, terminal `filled`). Ledger rows store the executed
child, so `events_from_order_rows` cannot recover the parent quantity from
a single fill row. That extractor is a best-effort reading; the session is
the stricter source.

`tests/formal/test_paper_conformance.py` runs `run_paper_loop` on a short
SYNTHETIC bar panel (research only; `live_pnl_claim` is false) and checks
that the order rows are an allowed trace with at least one fill.

## Accounting proofs

`quant_fund.formal.accounting` is the closed form the solver sees.
`tests/formal/test_accounting_smt.py` proves, over quantifier-free reals:

- Opening buy, partial close, flatten, long-to-short flip, and a three-trade
  unroll (buy, add, partial sell) satisfy
  `nav = initial_cash + realized + unrealized`, with
  `nav = cash + quantity * mark` and average-cost unrealized
  `quantity * (mark - average)`. Costs are expensed immediately into
  realized.
- `algebraic_total_cost` is non-negative and monotone in quantity and in a
  rate. Square-root impact is the real `s >= 0` with `s * s * adv = quantity * price`.
- A split preserves market value. A cash dividend moves cash and price so
  NAV is unchanged. The composition (split, then dividend on the new share
  count) preserves NAV.
- L1 turnover on three names is non-negative, zero on equal vectors,
  symmetric, and satisfies the triangle inequality.

On 2026-09-27, `pytest tests/formal/test_accounting_smt.py` reported each
proof call at about 0.01s (`--durations`). The solver returned `unsat` for
every negated claim. The whole `tests/formal` lane, including the stateful
machines, took 2.97s wall clock (`python3` `time.perf_counter` around
`pytest tests/formal -q`).

These proofs are about mathematical reals. The implementation is IEEE
float. A separate interior point
(`quantity=4`, `price=25`, `adv=100`, `sigma=0.05`, commission 1.5 bps,
half-spread 2.5 bps, `impact_y=0.2`, turnover 4 bps) checks that
`total_cost`, the closed form, and `fast_replay._order_costs` agree, and
that `financing_bps_per_year` does not change `total_cost`.

Hypothesis `AccountingMachine` (12 examples, 8 steps) buys and sells two
names with costs, marks, and restarts, and checks cash, quantities, NAV,
and `identity_gap` against the broker within `1e-6`.
`LifecycleMachine` (10 examples, 8 steps) rests limits, sweeps bars,
cancels, expires, and restarts, and requires every trace to be allowed
and never overfilled.

## Bugs

### Fixed

Same-ex-date split and dividend understated the total-return index.
`adjust_prices` used `dividend / previous raw close`. When a 2-for-1 and a dividend of 1 per post-split share share the ex-date,
raw closes 100 then 49, one pre-split share becomes two shares at 49 plus
2 cash (wealth 100). The split-adjusted
index should stay at 50. The old yield produced 50 then 49.5. The yield is
now `dividend * split_factor_prev / (previous raw close * split_factor)`.
When the factors match, that is still `dividend / previous raw close`.
A dividend on day 2 and a 2-for-1 on day 3 stays `[50, 55.5, 55.5]`.
Regression: `tests/regression/test_split_dividend_conservation.py`.
Default configs do not change a no-split dividend series.

Paper cash events omitted turnover that the broker already deducts.
`total_cost` includes `bps_per_turnover`. `record_orders` summed fee,
spread, and impact only. On a 1000 notional buy at 10 bps of turnover,
commission 1 bp and half-spread 5 bp, the cost total is 1.60 and the omitted
piece is 1.00. Slippage (decision price 99 versus fill 100, 10 of adverse
drift) is recorded and is not a cash charge. The fill now carries
`turnover_cost` (default 0) and the ledger adds it. Broker cash movement
is unchanged. `CostConfig.bps_per_turnover` defaults to 0, so runs that
leave it at 0 keep the same cash. Regression:
`tests/regression/test_cash_ledger_turnover.py`.

### Reported, not patched

Duplicate `submit` of the same `order_id` is not idempotent. Frictionless,
initial cash 100_000, two submits of 10 shares at 100: distinct fill ids,
20 shares, cash 98_000. The specification rejects the trace (`submit` is
enabled only from `absent`). The submission path is unchanged. Paper order
ids are fresh UUIDs, so the ordinary paper loop does not hit this.
Regression: `tests/regression/test_duplicate_order_submit.py`.

`run_backtest` deducts turnover inside `total`, and the reported cost
components sum commission, spread, and impact only. Adding a metric key
would change analytics-export digests, so the breakdown is unchanged.
`tests/regression/test_backtest_turnover_breakdown.py` pins the gap on a
flat SYNTHETIC book with `bps_per_turnover=25`: the NAV drop equals the
reported components plus turnover. Default rate 0 leaves sealed runs
alone. `fast_replay` has the same split: the fourth cost component
includes turnover, and the summed cost metrics do not.

`CostConfig.financing_bps_per_year` is validated and is not read by
`total_cost` or by execution. No financing charge was added. There is no
day-count in the cost function to implement one against, and inventing a
charge would move backtests.

## Limits

- One order in the TLA model. The validator is the interleaving product
  over order ids. Orders do not share state in the specification. Cash
  coupling is the accounting lane, not the state machine.
- Integer bounds in TLC. The Python twin accepts positive reals and
  arbitrary fill ids. They are not machine-checked equivalent.
- Liveness assumes no crash and no amend, plus weak fairness of resolve.
- Proofs are over reals, not IEEE float, and the trade proofs fix the
  branch structure (the symbolic sizes vary; the sequence of increase /
  reduce / flip does not).
- `events_from_order_rows` drops the parent quantity of an IOC-only row.
- A crash that discards an uncheckpointed fill is in the specification. The
  paper loop does not emit that trace.
- `financing_bps_per_year` is not part of the cost identity because the
  code does not charge it.
- Duplicate order ids and the backtest cost breakdown are known gaps. They
  are regression-tested and left in place so this change does not alter
  submission behavior or sealed digests.

## CI

`.github/workflows/ci.yml` job `formal` (timeout 10 minutes) installs
Temurin 21 and the locked environment, runs `scripts/run_tlc.sh`, then
`pytest tests/formal`. `tests/formal` is also on the default pytest
`testpaths`, so the lab suite collects it. The lab lint and examples jobs
on `main` are outside this lane.
