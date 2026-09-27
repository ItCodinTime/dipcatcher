# Forward shadow record (step 4)

`quant_fund.research.forward_shadow` records a frozen candidate and equal-weight
benchmark through future decision and execution windows. It uses step 3's
allocator and the same execution/accounting function as step 2's replay.

This is a local prospective simulation, not a live broker or an independently
verified track record. It does not fetch market data, schedule a worker or
manufacture a forward history. Supply real as-of observations as sessions occur.
The implementation tests use generated prices and controlled clocks only.

## Freeze before the first forward close

Copy `configs/forward_shadow.example.json` and replace every placeholder.
Choose the candidate before collecting any forward outcomes. A previously
selected tournament winner can be copied into `strategy`; cite the validation
receipt hash and run path in `selection_basis`. Manual preregistration is also
allowed when declared honestly. The benchmark, execution assumptions, asset
order, calendar, evidence design, code and runtime are frozen together.

The template deliberately has empty calibration observations and invalid date
placeholders. It cannot create a plausible-looking experiment without real
operator-supplied inputs. No code/runtime/clock override is exposed by the CLI.

```bash
PYTHONPATH=src python -m quant_fund.research.forward_shadow freeze \
  --spec configs/my_forward_protocol.json \
  --bootstrap inputs/bootstrap_closes.json \
  --run data/metadata/forward/shadow.sqlite
```

The database path is reserved exclusively. An existing run is never overwritten.
A failed initialization may leave an unusable reserved file; investigate it and
choose a new path, rather than reusing an ambiguous experiment identity.
Save the returned manifest and head hashes outside the working directory before
outcomes arrive. Later commands return a new journal head. Independent timestamp
attestation is an operational step; the code does not assert it has occurred.

## Input contract

`assets` is a fixed, unique, sorted list of security IDs. Every price/volume
vector uses that order. This is a fixed-universe experiment: record membership
selection bias in `selection_basis` and use verified consistent price/share
units. Corporate actions, dividends and real short locates remain unmodeled.

Bootstrap is a JSON list of at least `max(20, lookback, risk_window)` completed
close snapshots, with one close per UTC date and strictly increasing dates:

```json
{
  "event_time": "2026-09-23T20:00:00+00:00",
  "available_time": "2026-09-23T20:00:02+00:00",
  "close": [100.0, 101.0],
  "volume": [1000000.0, 900000.0]
}
```

Bootstrap observations must already be available when the protocol is frozen.
Each future session declares `signal_time` (completed close) and
`execution_time` (a later UTC date's open). Windows must be nonoverlapping and
future at freeze time. Supply the correct exchange calendar and all intended
sessions; the runner cannot independently verify calendar omissions or vendor
claims. All timestamps must carry a timezone and are normalized to UTC.

At each signal close, submit a JSON object with zero-based `session` and `bar`:

```json
{"session": 0, "bar": {
  "event_time": "2026-09-24T20:00:00+00:00",
  "available_time": "2026-09-24T20:00:02+00:00",
  "close": [101.0, 102.0], "volume": [1100000.0, 950000.0]
}}
```

```bash
PYTHONPATH=src python -m quant_fund.research.forward_shadow decide \
  --run data/metadata/forward/shadow.sqlite --input inputs/decision_0000.json
```

A decision must finish between its scheduled close and `execution_time` minus
the frozen `lead_seconds` margin. The deadline is checked again immediately
before transaction commit, after state reconstruction. This margin assumes the
filesystem completes its durable commit within the remaining lead time; the
local clock and storage are not externally attested. Both books' orders are saved
in one transaction before settlement is allowed. Orders include stable IDs and
signed share quantities; cost-aware decisions include allocator diagnostics.
Unavailable prices/volumes may be `null`; they cannot enter eligible signal
history. Missing a held asset's close blocks normal decisions.

Snapshots are archived as normalized JSON with a digest of the input request.
Previously archived closes are not revised. Availability is checked against the
local recording time, so this permits post-close feed latency within the decision
window. That differs from the tournament's strict close-time availability
protocol; do not claim identical signal timing on delayed vendor data.

After the scheduled open, submit its observed prices:

```json
{"session": 0,
 "event_time": "2026-09-25T13:30:00+00:00",
 "available_time": "2026-09-25T13:30:01+00:00",
 "prices": [101.5, 101.8]}
```

```bash
PYTHONPATH=src python -m quant_fund.research.forward_shadow settle \
  --run data/metadata/forward/shadow.sqlite --input inputs/open_0000.json
```

Settlement uses the previously committed quantities, frozen ADV/volatility and
cost assumptions. It logs fills, partial/unfilled quantities, rejects, borrow,
financing, cash, every position and marked NAV for both books atomically.
Unknown prices reject new orders; a missing held mark blocks the entire pair.
The next decision uses actual filled shares and cash. Partial quantities expire
at that session and are reconsidered by the next decision, not auto-filled.

The final scheduled decision requests liquidation using the same execution
participation caps. Residual positions remain marked and block final inference.
No emergency flatten or extra post-horizon trading is silently added.

## Missed windows and restarts

If no decision was committed before its deadline, submit the close snapshot
with `session` and a nonempty `reason` through `miss`, after the execution window
has elapsed. This records a missed decision for both books and holds existing
shares. Then settle that session's prices to account for market moves and carry.
It does not create retroactive orders, including on the terminal session.
A run with any missed decision is ineligible for the final statistical result.

Identical retries return the original event, even after the window closes.
Different inputs for the same committed decision/settlement fail. Failed valid
commands are logged where the journal is healthy; neither book moves on failure.
New sessions cannot skip an unsettled or unrecorded predecessor. A prolonged
outage therefore remains visible instead of disappearing from the return series.

The SQLite journal uses `synchronous=FULL`, serialized write transactions,
immutable-event triggers and a SHA-256 chain. Events and the derived state
projection commit together. Recovery validates the chain, reconstructs both
accounts by re-executing recorded orders against archived opens, and compares
cash, shares, fills, rejects and NAV with the persisted results. It checks that
the cached state matches this reconstruction before accepting another command.

```bash
PYTHONPATH=src python -m quant_fund.research.forward_shadow reconcile \
  --run data/metadata/forward/shadow.sqlite --output reports/shadow_review.json
```

The optional export is written exclusively and includes the manifest, journal,
account state, benchmark differences, overdue sessions and evidence readiness.
Use `--repair` to rebuild only a missing/mismatched derived projection. Journal
corruption or code/runtime changes cannot be repaired this way. To verify a
retained checkpoint exactly, pass `--expected-head SHA256`; a different head,
including a truncated record, fails that check. Preserve database backups using
SQLite-safe backup procedures and retain exported head digests independently.

The durability guarantees depend on SQLite's supported filesystem/OS behavior.
A local administrator can rewrite a whole database and its hashes or change the
system clock. A hash chain is not an external timestamp, vendor authentication
or tamper-proof evidence. Reports always disclose this distinction.

## Evidence length and final reporting

Supply pre-freeze **matched net-return differences** in `calibration`, with an
end timestamp and provenance. Declare the minimum economically meaningful daily
effect in basis points, a HAC lag, one-sided alpha and desired power. Calibration
must include at least `max(30, 4*(lag+1))` finite observations with nondegenerate
long-run variance. The planner estimates Bartlett/Newey-West long-run variance
and uses the fixed-sample normal approximation:

```
required_sessions = ceil((z_(1-alpha) + z_power)^2 * long_run_variance / effect^2)
```

The same minimum sample bound is applied. Positive dependence generally requires
more observations. This is an approximate design under stationarity and a normal
limit, not a promised power level under future regime changes. Calibration
provenance is declared by the operator, not independently verified.

A short calendar is allowed for operational testing but reports that it cannot
meet the evidence plan. Mean net differences remain descriptive while collecting
data. An approximate one-sided HAC test against zero is produced only after the
entire frozen schedule is settled, the planned sample size is met, neither book
has terminal positions and no session was missed. Repeated interim reports do
not produce interim p-values. Runs cannot extend their horizon after seeing
results. Multiple experiments still require external experiment tracking and
appropriate multiplicity control. Reports never authorize live promotion.

## References and checks

The local transaction model follows [SQLite atomic commit](https://www.sqlite.org/atomiccommit.html)
and [transaction isolation](https://sqlite.org/isolation.html). The Bartlett HAC
calculation follows the covariance formulation documented by
[statsmodels](https://www.statsmodels.org/stable/_modules/statsmodels/stats/sandwich_covariance.html).
The exact normal-approximation sample-size formula and assumptions above are
part of this protocol, not a claim that software can prove future profits.

Tests cover replay parity and independent cash reconstruction, idempotent and
concurrent retries, an actual subprocess exit before commit, projection repair,
corruption and retained-head checks, future/late inputs, missing marks, solver
integration, terminal residuals, dependence-sensitive planning and fixed-horizon
reporting. The original paper-loop API remains available; this forward protocol
specifically binds steps 2–3 to decisions committed before their outcomes.
