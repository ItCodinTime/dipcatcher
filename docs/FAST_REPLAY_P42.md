# `run_backtest_fast` — scope, contract, and gap analysis (P4.2)

Vectorized replay of the reference event loop for the matched-workload class.
The whole point of the path is that it is the *same* engine: outputs are
required to be **bit-identical**, and any workload outside the proven class
is refused fail-closed rather than approximated.

## How to select it

`run_backtest(bars, weights, config, *, fast=…)`:

| `fast` | behavior |
|---|---|
| `None` (default) | auto: runs the vectorized path only when it is semantically complete for the workload; otherwise the reference event loop |
| `True` | explicit contract: the vectorized path runs or the call raises `ValueError` — never silently degrades. This is the mode receipts should cite |
| `False` | pins the reference event loop (audit lane) |

`run_backtest_fast(...)` called directly enforces the same refusals — the
guards live in the callee, not just the dispatcher.

## Supported (proven bit-identical)

Target-percent weight panels (`event_time`, `security_id`, `target_weight`)
over daily bars where:

- `event_time` is `Datetime` in bars and weights **with the same unit**
  (mixed `us`/`ns` clocks would silently misalign weight cells — refused);
- bar `(event_time, security_id)` keys are unique (the matrices collapse
  duplicates — refused);
- the bars panel is non-empty;
- `execution.fill` is `NEXT_OPEN` or `CLOSE_AUCTION` with
  `allow_close_auction=False`;
- the full flat `CostConfig` surface (commission, half-spread, sqrt impact,
  turnover bps, borrow, participation limit, frictionless) and the full
  `RiskGateConfig` surface (order notional, gross/net/name/participation/
  predicted-vol caps, `stale_price_bars` fail-closed) — all enforced per
  order in the same order as the reference;
- kill switch (`HALT_NEW_ORDERS` counted per attempted order);
- shorts, missing bars, sparse rebalance grids (carried targets),
  delisted names, weight rows for names with no bars.

## Refused fail-closed (the gaps)

| workload class | why it is refused |
|---|---|
| `allow_close_auction=True` | close-auction order semantics are not replicated — a different fill-time model |
| `risk_overlay` (`BookRiskOverlay`) | overlay scales/flattens carried targets mid-loop; not replicated |
| `costs.spread_estimator` ≠ `flat` (Corwin–Schultz / Abdi–Ranaldo / Roll) | OHLC-calibrated half-spreads are only implemented on the event loop; the fast kernel would silently charge the flat floor — refused with a clear `ValueError` |
| GARCH / realized-GARCH market-overlay artifact present under `data.root/metadata/` | the per-order `max_predicted_vol` gate *is* replicated, but the `garch_risk_overlay_dates`/`realized_garch_risk_overlay_dates` metrics counters are stamped by the event loop and would silently read 0 — refused rather than approximate metrics. Closable: capture `market_risk_overlay_asof`'s source label per decision date and count it. |
| empty bars / duplicate bar keys / non-Datetime or mismatched-unit `event_time` | matrices have no faithful reading of these shapes |
| limit/stop order types | not in `run_backtest`'s contract at all (the API is target-percent weights only) — nothing to refuse |

## Byte-identity contract (why it is achievable)

Two subtleties had to be replicated exactly, and are where a future edit
would silently break identity:

1. **Summation order.** `Book.nav` and the exposure aggregates use CPython
   `sum()`, which on 3.12 applies Neumaier compensation over exact-`float`
   prefixes and degrades to a naive left fold at the first `np.float64`
   term. A participation-capped delta is `np.float64` (`np.sign`), so the
   fast path tracks per-share/per-scalar "np64" flags and replays the same
   prefix/fold structure (`_csum`/`_neumaier`). Aggregations in the metrics
   tail must use sequential `+=` folds — `np.sum`'s pairwise reduction
   differs at the last ulp (this burned `turnover_bps_cost` once; fixed).
2. **Contributor order.** Book NAV sums follow shares-dict insertion order;
   projected exposures follow sorted contributors. `_replay_kernel`
   reproduces both orders explicitly.

## Conformance evidence

- `tests/property/test_fast_replay_byte_identity.py` — hypothesis-generated
  workloads (seeds drawn from `st.integers`, `adversarial` profile); asserts
  `equity`/`fills` Arrow-IPC bytes identical, metrics digests identical,
  and the interpreted fallback identical to the numba kernel. Reference
  exceptions are required to reproduce with the same exception type.
- `tests/unit/backtest/test_fast_replay.py` — seeded 30-workload fuzz sweep
  plus targeted stale/close-auction/dup/kill/empty/sparse cases and flag
  behavior (`fast=True` refusals, `fast=False` pinning, auto-dispatch).
- `scripts/_conformance_11a.py` — the real 11-asset incumbent workload
  through both engines (remote diagnostic).

## Known issue discovered during this work (outside this lane)

`aggregate_shortfall` (`execution/implementation_shortfall.py`) orders
`top_cost_names`/`by_side` by `group_by` + `sort` on the aggregated value
alone; exact ties keep polars' hash-partition order, which is
**nondeterministic across executions** — the reference engine itself can emit
different orderings on identical input. Observed on a generated workload
with `total_is` ties of exactly 0.0. Fix belongs to that file (add a
secondary sort key, e.g. `security_id`); the conformance suite canonicalizes
element order in those two lists so content is still compared byte-exactly.
