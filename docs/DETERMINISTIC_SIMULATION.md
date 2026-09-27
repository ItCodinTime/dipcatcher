# Deterministic simulation testing

Library-only harness for replaying a research-to-paper session and swarming
seeded faults through it. The package is `quant_fund.simtest`. It drives the
real paper loop, the real `SimulatedBroker`, ledger and promotion-receipt
checks, and `run_backtest` on a labeled **SYNTHETIC** tape.

Nothing in this package submits to a live broker, opens a network socket, or
changes sealed receipts. Session output is a correctness trace. It is not a
research result and it does not carry a live-P&L claim.

## Seams

Two optional hooks keep the default paper path the same:

| Hook | Default | Simulation use |
| --- | --- | --- |
| `SimulatedBroker.id_factory` | `None` — fill ids stay `fill-{uuid4 hex[:12]}` and order ids stay `{slot}-{uuid4 hex[:10]}` | `DeterministicRuntime.mint` emits `sim-{kind}-{counter:08d}` |
| `run_paper_loop(..., prepare_broker=None)` | Brokers are constructed or restored and then used as before | Called after champion/shadow construct or `from_state`, before the first step. The session assigns `id_factory` and wraps `submit` |

`id_factory` is not persisted in `broker_state.json`.

One paper-loop change applies to every caller, including the default path:
`_validate_bar_panel` rejects a repeated `(event_time, security_id)` key.
Unique panels are unchanged. See [Bug: duplicate bars](#bug-duplicate-bars).

## Runtime

`DeterministicRuntime` owns a private `numpy` `PCG64(seed)` stream. Replay
does not re-roll that stream. `uniform`, id minting, feed delivery, clock
readings, cache IO, and the simulated research API all go through
`effect()`, which appends a request/response pair.

State commits hash the canonical JSON of each intermediate payload with
SHA-256. Replay recomputes the hash and raises `ReplayDivergence` if the
call sequence or the hash differs.

The event log is a compact binary:

- magic `DST1`, version 1
- zlib level 9
- canonical JSON payloads
- floats as IEEE hex (`{"$f": float.hex()}`), datetimes as `{"$t": isoformat}`, bytes as `{"$b": hex}`

A recording run and a second recording run produce the same compressed bytes
and the same list of state hashes. Replaying those bytes reproduces every
hash, plus cash, shares, order ids, and fill ids. The proof is
`tests/unit/simtest/test_replay.py`.

Fault schedules use a separate stream, `PCG64(seed ^ 0xD1570001)`, so the
fault list is an input. Replay receives the same `FaultSchedule` and does
not draw it again.

## Session

`run_session` builds a three-name tape (`AAA`, `BBB`, `CCC`) from Monday
2026-03-02, weekday closes at 21:00 UTC. That calendar crosses the 2026 US
spring-forward. Opening bars are the first close and carry no feed faults.
Step `k` delivers execution bar `k+1`. Day-one weights are a fixed `AAA`
long of `0.04`; later weights are the sign of the last close-to-close move
at `±0.04`. Shadow weights are half of that. Initial cash is `1_000_000`.

The paper config is `configs/paper.yaml` with a temporary `data.root`,
`source: synthetic`, runtime mode `PAPER`, ledger subdir `simtest-paper`,
`promote_min_steps=5`, `stale_price_bars=8`, loosened risk limits, and
`costs.participation_limit=1.0` so a partial-fill fault is the only size
cap. `run_id` is `sim{seed:08x}`.

Each step, in order: advance the simulated clock, deliver and normalize
bars, write and read a signal cache, call `sim://research/signal`, then
`run_paper_loop` for one new decision (`prefer_latest=False`, resume after
the first checkpoint). A fail-closed step stops the session. Later days are
not traded on a torn calendar.

After a checkpoint, one extra resume with no new dates must leave cash,
shares, order ids, and fill ids unchanged. `run_backtest` then consumes the
same normalized tape. A `StaleValuationError` records backtest
`fail_closed`. Broker faults apply only to `SimulatedBroker`, not to the
backtest.

The GARCH overlay looks for `vol_garch.joblib` under `data.root`. The
temporary root has no artifact, so the overlay stays off and the session
does not load a model.

Outcomes are `ok`, `fail_closed`, or `recovered`. The hash-count invariant
is `len(state_hashes) == len(outcomes) + (1 if backtest != "not_run" else 0)`.
The resume commit is one of the outcomes. The backtest commit is the extra
hash.

## Clock

`SimClock` does not read the wall clock. Exchange time (`true_time`) only
moves forward. Session time is `true_time + offset`. Bars are released by
calendar index, so a fast clock cannot see a future bar. If session time is
behind the exchange target, the step is `causal_block` and the session
fail-closes.

| Fault | Effect |
| --- | --- |
| `clock_skew`, `clock_jump` | Add `magnitude` seconds to the offset. A large negative magnitude puts session time behind the exchange and fail-closes |
| `clock_dst` | If the step crosses a real `America/New_York` transition, apply that UTC-offset delta. Spring-forward 2026-03-08 is `+3600` s (EST −18000 to EDT −14400). Fall-back around 2026-11-01 is `−3600` s and sets `repeat=True`. If the step has no civil transition, inject a synthetic `+1` hour so the fault is still observable |
| `clock_leap` | Python `datetime` cannot represent `23:59:60`. The fault repeats the previous UTC instant and sets `leap_second=True`. A repeat records `recovered` and the day still trades once |

## Feed, cache, and API

| Fault | Effect |
| --- | --- |
| `feed_gap` | Drop the target name from that delivery |
| `feed_duplicate` | Append the target bar. `conflict=True` scales prices by `1.01`. Exact duplicates are dropped. Disagreeing OHLCV returns status `conflict`, fail-closes, and stops the session. Normalization is order-independent |
| `feed_reorder` | Reverse the day's rows, then sort. Economics match the unshuffled day |
| `disk_full` | `SimDisk.write` raises `DiskFull` (`errno` 28) before committing bytes. The step fail-closes |
| `corrupt_cache` | Flip the first byte on read. A digest mismatch rewrites the clean bytes and records `recovered` |
| `api_slow` | `sim://research/signal` still returns HTTP 200. Latency is `5_000_000` µs instead of `100` µs |
| `api_5xx` | HTTP 503 body `{"ok":false}`. The step fail-closes and the session stops |

`SimDisk` and `SimNetwork` are in-memory. The paper ledger still uses the
host filesystem inside a temporary directory.

## Simulated broker faults

These faults run only inside the session's submit wrapper. `SimulatedBroker.submit`
itself is unchanged for every other caller.

Precedence on a step is `crash`, then `broker_timeout`, then `broker_reject`,
then `partial_fill`. Crash and timeout raise on attempt 1 **before** the
fill. The session retries once with the fault disarmed. The paper loop
checkpoints only after a finished step, so the crashed attempt is discarded
and the retry resumes from the last durable cursor. A reject returns an
`OrderRecord` with `reject_reason=sim_broker_reject`, appends that record to
the simulated broker history, increments `reject_count`, and does not move
cash.
A partial fill scales quantity by `magnitude` clamped to `[0.05, 0.95]`
before the real submit. If history already holds that `order_id` with a fill
or a reject reason, the wrapper returns that record.

A day with no orders does not exercise these faults. Day 0 buys `AAA`.

## Invariants

`check_invariants` requires all of the following:

- `research_only` is true, `live_pnl_claim` is false, `data_source` is `SYNTHETIC`
- every outcome is `ok`, `fail_closed`, or `recovered`
- the extra resume is idempotent
- promotion-receipt errors are empty when a checkpoint exists
- ledger schema is ok when cash is present
- cash and champion shares rebuild from fills (signed quantity × price, plus fee, spread, and impact) and from champion `borrow` rows in `cash_ledger.parquet`, within `1e-4` cash and `1e-6` shares
- shadow cash is about 0
- fill ids are unique, and filled order ids are unique
- the state-hash count matches the formula above

No Sharpe, Sortino, Calmar, or P&L headline is part of the swarm report.

## Swarm and shrinking

`schedule_from_seed(seed, n_steps, max_faults=3)` draws `0 .. max_faults`
faults. `partial_fill` magnitude is `Uniform(0.2, 0.8)`. Other magnitudes
are seconds `Uniform(-3 days, +3 days)`. `conflict` is Bernoulli `0.5`.
The list is sorted so the schedule is stable.

`run_swarm` runs seeds `base_seed .. base_seed + n_seeds - 1`. An exception
or a failed invariant is a swarm failure. `shrink_schedule` is ddmin: it
returns a subset-minimal schedule that still fails, or the empty schedule
when the empty schedule fails on its own. Shrinking a passing schedule
raises `ValueError`. An unexpected exception during a shrink candidate
still counts as failing.

```bash
make simtest          # unit + regression, then 64 seeds × 5 days
make simtest-large    # 4000 seeds × 8 days
uv run python scripts/simtest_swarm.py --seeds 48 --days 5
```

`tests/unit/simtest/test_swarm_slow.py` is marked `slow` and runs 16 seeds.
The default lab pytest job includes slow tests (`-m "not network"`), so that
test stays in the main suite. The simtest workflow's own pytest step skips
it and runs the 64-seed swarm instead.

## CI

`.github/workflows/simtest.yml`:

- every pull request: non-slow simtest unit tests, the duplicate-bar
  regression, and a 64-seed × 5-day swarm (20 minute job timeout)
- `workflow_dispatch`: large swarm, default 4000 seeds × 8 days (90 minute
  job timeout). Inputs override `seeds`, `days`, and `base_seed`

There is no cron. The large job runs when dispatched.

## Measured results

All figures below are from this host (`uv run python`, CPython 3.12,
`MLFLOW_DISABLE_AGENT_HINT=1`). They are infrastructure measurements on a
SYNTHETIC tape. They are not model scores.

### Swarm, 200 seeds

Command: `scripts/simtest_swarm.py --seeds 200 --days 5 --base-seed 0`
(default `max_faults=3`, shrinking enabled). Exit code 0.

| Field | Value |
| --- | --- |
| elapsed_seconds | 101.77519267599996 |
| sessions_per_second | 1.9651154150766137 |
| failures | 0 |
| log_bytes_min | 805 |
| log_bytes_max | 2486 |
| outcomes `ok` | 978 |
| outcomes `recovered` | 67 |
| outcomes `fail_closed` | 55 |

The same command with `--seeds 64` (the pull-request job) took
29.110639300999992 s, 2.198508914155022 sessions/s, 0 failures, log bytes
867–2474, and outcome counts `ok` 306, `recovered` 19, `fail_closed` 18.

A 4000-seed × 8-day run was not executed here. At the 200-seed rate above,
4000 sessions of 5 days would be about 34 minutes before accounting for
8-day sessions and a slower CI runner. The workflow timeout is 90 minutes.
Do not treat that extrapolation as a measured 4000-seed result.

### Replay

`test_record_twice_and_replay_share_every_state_hash` records a 4-day clean
session twice and replays the first log. The three runs share every state
hash, the compressed log bytes, cash, shares, order ids, and fill ids.

### Duplicate-bar economics (unique panels)

These two runs are single, unambiguous fill bars. They show the two books
the old last-row-wins map could select. Methodology: frictionless
`configs/paper.yaml`, one name `A`, decision 2024-01-02, fill 2024-01-03,
tail 2024-01-04, target weight `0.5`, initial NAV `1_000_000`, close `100`
on every row, `max_steps=1`.

| Fill open | Cash | Marked NAV | Fill quantity | Fill price |
| --- | --- | --- | --- | --- |
| 100 | 500000.0 | 1000000.0 | 5000.0 | 100.0 |
| 110 | 500000.00000000006 | 954545.4545454546 | 4545.454545454545 | 110.0 |

Frictionless cash stays near half of the initial NAV because notional equals
target weight times NAV. The order-dependent damage is the share count and
the marked NAV.

## Bug: duplicate bars

`_bar_maps` walks a day and keeps the last row for each `security_id`. A
frame with two rows for the same `(event_time, security_id)` therefore
filled at whichever open arrived last. Reordering `[100, 110]` versus
`[110, 100]` on the fill bar selected the two books in the table above.

The minimized regression is
`tests/regression/test_simtest_duplicate_bar_order.py`:

- both orders of conflicting opens raise
  `ValueError` matching `duplicate bars`
- `normalize_bars` reports `conflict` independent of input order
- a session whose step-0 fault is `feed_duplicate` with `conflict=True`
  records `fail_closed`, leaves cash unset, and passes the invariants

The backtest engine already rejected duplicate target-weight keys. The paper
loop now rejects duplicate bar keys the same way. Exact duplicates are
rejected too: the panel is ambiguous.

No other invariant failure showed up in the 200-seed swarm, so there is no
second minimized bug in this pass.

## Limitations

- A fail-closed day ends the session. The harness does not skip that day
  and keep trading.
- An earlier `api_5xx`, `disk_full`, or conflicting duplicate stops the
  session, so a later fault on the same schedule never runs. With
  `max_faults=3`, many seeds have zero or one fault. Stacked faults are
  under-sampled.
- Crash and timeout raise before the fill. They do not model a durable fill
  whose acknowledgement was lost. A retry of an `order_id` already in
  history with a fill or a reject reason returns that record. A torn cursor
  (equity rows ahead of the checkpoint) still fail-closes in the existing
  resume check.
- `load_broker_state` maps corrupt JSON to `None`. Resume then reports the
  state file as missing. That message is unchanged.
- A leap second is a repeated UTC timestamp, not a civil `23:59:60`.
- Broker faults fire only when `submit` is called.
- Paper-loop NAV and backtest NAV are not required to match. The backtest
  is a second consumer of the normalized tape.
- The simulated disk and HTTP client are not the paper ledger's filesystem
  and are not a vendor HTTP adapter.
- This host did not run the 4000-seed swarm.
