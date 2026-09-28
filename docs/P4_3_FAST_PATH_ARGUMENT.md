# P4.3 — Why the event loop cannot reach fast-path latency honestly

ULTRAPLAN P4.3 asks: if the reference event loop can't reach ≤1× of the
vectorized path honestly, write the argument. This is that argument, with
the profile evidence from `docs/PERF_SWEEP.md` (P4.1/P6.8 sweep).

**Verdict: the residual gap is the product, not overhead.** Per-order
fail-closed risk gates, cost functions, and order-schema validation are
sequential by contract — a vectorized reducer without them is a different
engine. The honest fix is not a faster event loop; it is a second
implementation of the *same* semantics, proven byte-identical (P4.2).

## Latency decomposition (11 assets × 1000 days, SYNTHETIC)

Reference event loop after the P6.8 bit-identical sweep (1.41× speedup:
166.2 ms → 117.8 ms median, 1.41M → 887k cProfile calls). Residual vs the
vectorized replay (~22 ms) is ≈5.3×, decomposed:

| Share | Where | Why it can't leave |
|---|---|---|
| ~80% | `total_cost`, `check_order`, `_make_order` + pydantic `Order` validation, `sqrt_impact`, `_projected_exposures` | Sequential book dependency: each order mutates shares/cash consumed by the next; per-call fail-closed validation is the contract. |
| ~15% | Interpreter loop body: dict bookkeeping, `sorted(ids)`, nav/turnover accounting | CPython dispatch; numba-izing the loop body *is* the fast path. |
| ~5% | Polars marshalling residue, `_build_result` | I/O boundary; already minimized. |

A "vectorbt-class" backtester wins by deleting the sequential contract:
orders become a weight matrix times a price matrix, and gates become
ex-post filters. dipcatcher's engine evaluates every order against the book
state *as mutated by prior orders*, the kill switch, stale-mark staleness,
and the six risk gates — in iteration order. That is the semantic being
sold; removing it for speed is dishonest by definition.

## What the fast path buys instead of speed-for-semantics

P4.2 makes `run_backtest_fast` the *same engine*, not a fast approximation:

- Byte-identical `equity`/`fills` Arrow-IPC frames and metric digests on
  generated workloads (`tests/property/test_fast_replay_byte_identity.py`),
  including a seeded 100-workload sweep and the interpreted fallback.
- Refusals are fail-closed and enumerated (`docs/FAST_REPLAY_P42.md`):
  close-auction, risk overlays, GARCH artifacts, malformed panels →
  `ValueError`, never silent degradation.
- Fault-injection parity: kill-switch halts and `StaleValuationError`s
  reproduce with identical exception types and args — the conformance
  suites' wins are on *failure behavior*, not just happy paths.

## The rejected levers (and why)

From the P6.8 sweep, each rejected for breaking provable identity:

- Vectorizing per-order cost/gate calls — breaks the sequential book
  dependency and the per-call validation contract.
- `np.sum`-style pairwise reductions for NAV/exposures — pairwise
  summation differs from CPython `sum()` at the last ulp (the
  Neumaier-compensated-then-naive fold structure is replicated instead).
- Cross-order batching — changes fill ordering.

## Consequence for P4.7

The industry-grade claim for this lane is *not* "as fast as a vectorized
reducer" — it is "a vectorized latency profile **with** per-order
fail-closed semantics, proven byte-identical to the audited reference,
with refusal semantics enumerated." Any further event-loop speedup must
remain in the provably-bit-identical set; the residual ~5.3× is the price
of the contract and is accepted.
