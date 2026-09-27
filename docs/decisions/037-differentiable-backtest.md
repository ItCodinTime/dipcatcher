# ADR-037: Differentiable research backtest

## Status

Accepted

## Date

2026-09-27

## Context

The lab can score a research book in NumPy (directional TSMOM, top-k,
Antonacci, inverse-vol, and the rank weights in `net_replay`) but cannot
differentiate that book with respect to its parameters or the price path.
Grid search on a handful of knobs does not say whether a selected point is a
spike. A separate effort is splitting large modules, typing the public API,
and adding an event-driven simulator. This work does not touch those paths,
live trading, or sealed receipts.

## Decision

1. New package `quant_fund.diffbacktest`. The NumPy core does not import JAX.
   JAX is an optional extra, CPU-only in CI.
2. Hard mode matches the existing books (band off, impact off) within
   `1e-8`. Smooth mode is a softplus / soft-threshold / soft top-k relaxation.
   STE uses hard trades on the forward pass and the smooth surrogate on the
   backward pass.
3. Out-of-sample comparison is walk-forward on the hard book. The flat
   objective is sample ratio minus λ times the smooth gradient norm, on the
   same free parameters as the grid. The synthetic small case is fixed in
   `spec.py` and is not retuned after the numbers are known.
4. The adversarial radius is an L∞ ball in frozen full-sample volatility
   units. It is an upper bound from projected gradient plus a NumPy
   certification, not a robustness certificate.
5. Reports carry `research_only=True`, `live_pnl_claim=False`, and an explicit
   data-source label. The small case is `SYNTHETIC`.

## Consequences

- `make diffbacktest` and the `diffbacktest` CI job run the new tests on CPU.
- Coverage of the new package counts toward the existing 80% floor because
  the main test job installs every extra.
- The container image stays free of JAX.
- No broker, order-router, paper-loop, or receipt change.
