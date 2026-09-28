# ADR-0005: Ledger artifacts are durable before the resume cursor advances

## Status

Accepted (discovered; documents existing behavior)

## Context

`run_paper_loop` supports multi-day resume via `broker_state.json` under
`data/metadata/paper/<run_id>/`. A crash mid-run creates a classic
durability hazard: if the resume cursor (last processed decision/exec
date) is persisted before the accounting rows it summarizes, a later resume
would skip bars whose ledger entries were never written — silently losing
fills.

The loop's ordering (`paper/loop.py`, per decision date):

1. `ledger.record_orders` / `record_snapshot` / `record_shadow_equity`
2. `ledger.flush()` — atomic temp+fsync+`os.replace` per artifact
   (`paper/ledger.py` `_atomic_write_text/_atomic_write_parquet`)
3. `ledger.save_broker_state(...)` — the resume cursor written **last**

At run end the same discipline repeats: all ledgers and `meta` flush first,
then `broker_state.json` is published "intentionally last, after every
ledger artifact has been durably replaced" (inline comment in `loop.py`).

Resume is additionally bound by `_paper_resume_fingerprint`: a SHA-256 over
`config.dump()` + the already-processed bar prefix (sorted columns, row
hashes). A mismatched fingerprint raises — "refusing to replay altered
evidence". Restored marks restart at age zero and age deterministically
(`_bounded_valuation_marks`), and prior `mark_ages`/step/cursors are
type-checked on load.

## Decision

The resume cursor is published **after** every durable artifact it
summarizes, and resume only proceeds when inputs hash-identical to the
prior run's prefix are supplied.

## Consequences

- A crash can lose the *cursor* (harmless — the step replays) but never
  leaves a cursor claiming progress the ledger lacks.
- Resuming against edited bars/config is a hard error, not a divergence
  warning — replay integrity equals receipt integrity.
- `equity.parquet` row count must match the broker cursor, and a legacy
  state file missing fields fails validation rather than being coerced.
- Cost: extra fsyncs per step; acceptable at paper-loop cadence.
