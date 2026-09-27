# ADR-0001: Research receipts are immutable, self-sealing evidence

## Status

Accepted (discovered; documents existing behavior)

## Context

The lab's central claim is *receipt-bound research*: a benchmark or
tournament result must be verifiable long after it was produced, including
negative results (e.g. `selected_band: null`, `development_eligible: false`
in `receipts/`). Research claims otherwise degrade into unverifiable
narrative, and a mutable "latest" artifact can be silently edited to look
better.

The implementation is spread over three sites:

- `src/quant_fund/research/agent.py` — `run_research` writes per-run
  `runs/<run_id>.{json,md}` plus `latest.{json,md}` through
  `_atomic_write_text` (tempfile + `os.fsync` + `os.replace`), then binds
  `immutable_json_sha256` over the canonical payload.
- `src/quant_fund/research/verify.py` — `_receipt_digest` recomputes the
  SHA-256 with the self-referential digest field excluded;
  `verify_research_artifact` fails closed on any mismatch, missing
  provenance field, or out-of-range statistic.
- `src/quant_fund/paper/ledger.py` — the same atomic-write discipline and a
  self-excluding `receipt_sha256` for `promotion_dry_run.json`.

## Decision

Receipts are **write-once artifacts sealed with a self-excluding digest and
published atomically**. A receipt is either byte-identical to what the
pipeline produced or invalid; there is no legitimate "edit" path. Verification
recomputes the digest rather than trusting the file, and CI
(`ci.yml` "Validate research artifact") exercises `verify-research` on every
run.

## Consequences

- A failed write cannot replace a previously valid receipt with partial JSON
  (atomic rename in the same directory).
- Hash binding is canonical (`json.dumps(..., sort_keys=True,
  separators=(",", ":"))`), so verification is platform-stable.
- "Blocked"/negative outcomes remain reviewable forever — a sealed blocked
  tournament stays `valid: true` evidence (`docs/RECEIPT_VERIFICATION.md`).
- Cost: any schema change to a receipt shape must be versioned
  (`RESEARCH_RECEIPT_SCHEMA_VERSION`, `BENCHMARK_CATALOG_VERSION`) or
  existing receipts stop verifying.
- Promotion evidence must be bound to a verified receipt: candidate metrics
  must match `provenance.run_id` **and** the current
  `git_worktree_sha256` (`validation/gates.py`).
