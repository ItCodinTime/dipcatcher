# research/reality — trial ledger lifecycle

This directory holds the **pending** reality-filter study plus an archive of
decided studies.

## Pending (this directory's top level)

- `preregistration.json` — the frozen spec for the study currently under
  evaluation. Written before any result is computed; the runner refuses a
  config that differs from it.
- `trials.jsonl` — the committed ledger export scored by `make reality-gate`
  (`COMMITTED_TRIAL_LEDGER`). While it is non-empty the CI `reality-filter`
  job scores it with the unchanged filter and fails on any verdict other
  than `pass`.

## Decided (`studies/<study_id>/`)

When a study's verdict is rendered and published (`docs/REALITY_*.md`,
sealed `receipt.json`, hash-chained `audit/`), its artifacts move to
`studies/<study_id>/` and the pending paths clear. This is required, not
optional: `run_sweep` refuses to record a second study while a previous
one's audit ledger occupies the default paths (`audit ledger already
exists; refusing to re-sign`), and `trials.jsonl` is a per-run export
(`open("w")`) — a decided study's rows must not contaminate the next
study's verdict.

Nothing is deleted on archival: the ledger rows, frozen spec, return panel,
sealed receipt, and audit chain stay byte-identical under `studies/`.

## Studies

| study | verdict | archive |
|---|---|---|
| `reality-us-liquid-daily-2026-09-27` | `deflated` (DSR 0.684 < 0.95, PSR 0.801, 29 trials, 3 clusters) | `studies/reality-us-liquid-daily-2026-09-27/` — see `docs/REALITY_TRIAL_2026.md` |

The generated index of every recorded batch is
[`docs/research/findings.md`](../../docs/research/findings.md). Regenerate it
with `uv run python scripts/research_findings.py`. It copies figures from
artifacts and does not fill a missing one.
