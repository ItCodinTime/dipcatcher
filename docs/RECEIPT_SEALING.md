# Receipt sealing conventions

Every durable JSON artifact under `src/quant_fund` is either self-sealed or
explicitly whitelisted. This page is the policy the seal-coverage ratchet
(`tests/unit/test_evidence_seal_coverage.py`) enforces mechanically.

## Convention

A sealed artifact carries a top-level `receipt_sha256` field equal to the
SHA-256 of the artifact's other fields serialized canonically:

```python
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

body = {...}  # must not contain "receipt_sha256"
sealed = {**body, "receipt_sha256": hash_bytes(canonical_json_bytes(body))}
```

`canonical_json_bytes` is total: non-finite floats become `null`, unknown
types `str()`-ify, keys are sorted, separators compact. The seal therefore
round-trips — a verifier recomputes `hash_bytes(canonical_json_bytes(
{all fields except receipt_sha256}))` from the parsed JSON and compares.

When re-sealing an object that may already carry a seal, strip the stale
field first — hashing a payload that still contains `receipt_sha256`
produces a seal-of-seal that no verifier will accept:

```python
body = {k: v for k, v in payload.items() if k != "receipt_sha256"}
```

`research.receipt_v2.seal_receipt` does this for you; data-layer modules
(which must not import `research/`) inline the three lines above.
Domain-specific digests (`artifact_sha256`, `report_sha256`) follow the same
self-exclusion rule and count as seals.

## Writer policy

- **Durable evidence** — receipts, manifests, provenance sidecars, checkpoint
  manifests, report artifacts under `data/`/`receipts/`/`artifacts/` or any
  `--out` path — must seal before writing.
- **Non-evidence** — stdout-echoing CLI emitters, transport payloads, and
  snapshots that merely echo a tracked input — are whitelisted in
  `UNSEALED_WRITERS` with a justification.
- A new JSON writer that is neither sealed nor whitelisted fails the
  ratchet test.

## Verification

`dipcatcher verify-receipt <path>` recomputes the seal for every sealed
payload and additionally deep-verifies known kinds: `receipt.v2` envelopes
(pydantic schema, environment fingerprint, code-map digest),
`fleet_eval.v1` contracts, data manifests (`schema_version: 1` +
`artifacts`), and kind-tagged lane receipts. `doctor` re-verifies the data
manifest's per-artifact digests on disk *and* its seal.

`mc_engine` checkpoint manifests are seal-checked on resume: a manifest
whose seal does not match its body refuses to load. Pre-seal checkpoints
without the field still load — they are working state, not published
evidence.

## Determinism

Receipt content is required to be byte-identical across processes.
`PYTHONHASHSEED` is read at interpreter startup, so hash-order leaks only
appear across subprocesses — `tests/unit/determinism/` runs sim_live, the
gold pipeline, and fleet_eval under different seeds and compares the sealed
payloads minus wall-clock audit fields (`generated_at`, `ingested_time`,
path roots).
