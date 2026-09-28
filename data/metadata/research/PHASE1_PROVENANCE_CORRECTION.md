# Phase-1 provenance correction

## What was corrected

#169 (`7d2e01e4dba09d178ed46ad802950d2296d3eae6`) replaced the
2026-09-25 tournament code maps, changed the recorded cvxpy version from
`1.9.2` to `1.9.3`, and recomputed the receipt links. The studies were not
re-run. A later code review or library upgrade cannot establish which code
and runtime produced an earlier result.

The original sealed files are restored byte-for-byte from the parent of
#169, `b35429bb52de618dbda42d354aa40f142e104c57`. This restores the
two tournament manifests, three compressed result receipts, their attempt
receipts, and the Phase-1 index. It also restores the original forward-shadow
index pin and the regression tests that protect these bytes. Later evidence
page additions are preserved; only the superseded provenance references
are corrected.

## Historical execution provenance

- Index Git revision: `c564646b14849d72c5891a90034d8e26361fe0f5`.
- Index file SHA-256: `3e8d28e2ad08137d5e0ea7b8703600d710cdda37ebe83e384afc50fbc94d1f7a`.
- Index embedded receipt SHA-256: `0ce794b56249952fce5b2ff1046eea9e50b2f4e6d691539b8019959131873204`.
- Tournament runtime cvxpy stamp: `1.9.2`.
- Net-tournament manifest receipt: `4650484a42ea0bbb926e46f24f1d85dc46016ad87585523dd5e26380e8d3bf13`.
- Cost-aware manifest receipt: `7dab6ee1eb6b39c47e85f048c8f9584d5c6b08f214ef3c6a4cb908228be7ca90`.

The original timestamps, result fields, code maps, runtime stamps, and hash
links are immutable historical evidence. They do not establish results under
the current source tree or current environment. Checking their hashes and
links verifies the recorded bytes; it does not reproduce an experiment.

## Superseded reseal

The [earlier reseal account](PHASE1_CODE_RESEAL.md) remains available for
audit. Its replacement index receipt
`bfce88b1efdccc1b15a4e925de79ad0ef10c0fd33102744c08f72374b08fb8dd`
and replacement revision `5c4e0c876f6d5f74e42d1205468a8768752e14e2`
are superseded provenance, not a new execution record.

A reproduction under changed code or dependencies must create a new run
directory and new receipts that link back to these historical artifacts.
It must disclose the reused, previously inspected holdout. Source drift or
lock-file verification limitations must be fixed in the verifier or recorded
as limitations; they must not be repaired by rewriting execution stamps.

All evidence remains retrospective research evidence. No new market test,
live-trading evidence, promotion, or performance result is claimed here.
