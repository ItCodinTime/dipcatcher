# ADR-0004: Degraded inputs fail closed — never emit plausible-looking output

## Status

Accepted (discovered; documents existing behavior)

## Context

A research engine whose output feeds receipts must never fabricate
believable numbers from bad inputs. The codebase applies one rule
everywhere: when an input or precondition is invalid, raise (or mark the
artifact invalid) instead of substituting a default that would pass as real.

Representative sites:

- `backtest/engine.py` `StaleValuationError` — a held position whose last
  mark ages past `risk_gate.stale_price_bars` stops the run rather than
  emitting fabricated NAV/P&L. Same error type in `paper/loop.py`.
- `portfolio/risk_gate.py` `check_order` — non-finite inputs, non-positive
  NAV/price, stale price/model age all raise `RiskGateRejected`.
- `monitoring/kill_switch.py` — any state other than `ENABLED` (including
  *unknown*) blocks new orders; `may_flatten` requires
  `human_authorized=True` even when `allow_auto_flatten` is set.
- `config/models.py` — `allow_live: true` always raises
  ("no live broker adapter in this repository"); unknown `data.source`,
  unimplemented `optimizer.covariance` names, and unsafe
  `paper.ledger_subdir` paths all fail at validation.
- `data/ingest.py` `make_provider`, `pipeline/dataset.py` `panel`,
  `pipeline/forecast.py` `forecast_asof` — unknown sources, stale/feature-
  mismatched cached gold, and missing as-of rows all raise instead of
  falling back ("refusing latest-date fallback").
- `validation/gates.py` — a missing/invalid notebook or metrics file is a
  failed gate, not a skip; `verify_research_artifact` treats absent
  provenance as invalid.

## Decision

Default to **fail closed**: invalid input → exception/invalid artifact;
absent capability → refusal. Where a heuristic fallback exists (e.g. the
`cs_pct_mom_20` momentum heuristic in `forecast_asof`), it applies only
when no trained artifact was loaded at all — a loaded-but-mismatched
artifact raises.

## Consequences

- Silent degradation is structurally impossible on the gates that matter
  (valuation, orders, promotion, live claims).
- Tests assert the *absence* of output on bad input, not just presence on
  good input (`tests/unit/pipeline/test_*fail_closed*`).
- Cost: operational tooling must handle exceptions as first-class outcomes
  (reject accounting `reject_total`, `risk_gate_rejects` in metrics rather
  than swallowed errors).
