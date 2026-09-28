# 038 — Reality-ledger lifecycle: pending vs decided studies

`research/reality/trials.jsonl` is the *pending* adjudication input to
`make reality-gate`, not a cumulative history. Three facts fix this
semantics:

1. `run_sweep` writes the ledger with `open("w")` — it is a per-run export
   of the study under evaluation, so rows from a decided study would
   contaminate the next study's verdict if left in place.
2. `run_sweep` refuses to record a second study while a previous audit
   ledger occupies the default paths ("audit ledger already exists;
   refusing to re-sign with a new key").
3. The workflow already defines the empty/absent case as a noticed skip —
   the designed steady-state between studies.

Therefore a decided study — verdict rendered, published in
`docs/REALITY_*.md`, sealed in `receipt.json`, hash-chained in `audit/` —
moves byte-for-byte to `research/reality/studies/<study_id>/`. Nothing is
deleted and nothing is edited: the negative verdict stays published, the
ledger rows stay committed, and the gate still fails closed on any future
pending ledger whose verdict is not `pass`.

First application: `reality-us-liquid-daily-2026-09-27` (verdict:
`deflated`, docs/REALITY_TRIAL_2026.md) archived to
`research/reality/studies/reality-us-liquid-daily-2026-09-27/`.
