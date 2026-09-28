# receipt.v2 — unified sealed receipt envelope

`receipt.v2` (ULTRAPLAN P7.2 + P7.4) is the single envelope every research
lane — eval, incumbent, carry, paper — seals evidence into. The published
contract is `src/quant_fund/research/receipt_v2.schema.json` (JSON Schema
2020-12); `quant_fund.research.receipt_v2.ReceiptV2` is its executable form
and the verifier's structural check. `tests/unit/research/test_receipt_v2.py`
keeps the two in sync.

## Envelope

```json
{
  "schema": "receipt.v2",
  "schema_version": 2,
  "kind": "distribution_fleet_eval",
  "data_label": "SYNTHETIC",
  "generated_at": "<iso-8601 with tz>",
  "git_revision": "<commit or UNKNOWN>",
  "dataset_hash": "<sha256 of canonical dataset identity>",
  "params_hash": "<sha256 of canonical run params>",
  "code_sha256": "<sha256 of canonical {filename: sha256} map>",
  "code_files": {"fleet_eval.py": "<sha256>"},
  "environment": {
    "python": "3.12.x", "implementation": "CPython",
    "platform": "...", "machine": "...", "byteorder": "little",
    "packages": {"numpy": "...", "polars": "...", "scipy": "..."},
    "blas": {"name": "accelerate", "found": true, "version": null},
    "lapack": {"name": "accelerate", "found": true, "version": null},
    "threadpools": [{"prefix": "libopenblas", "user_api": "blas",
                     "internal_api": "openblas", "num_threads": 8, "...": "..."}],
    "fingerprint_sha256": "<sha256 of every other environment field>"
  },
  "live_pnl_claim": false,
  "verdict": "pass",
  "payload": {"...": "lane-specific receipt body (e.g. fleet_eval.v1 verbatim)"},
  "receipt_sha256": "<sha256 of canonical JSON of every other top-level field>"
}
```

- `verdict` is `pass` | `fail` | `blocked`. For the fleet lane it is `pass`
  iff zero error rows; a recorded head failure verdicts `fail`.
- `environment` is the P7.4 per-receipt numeric fingerprint: interpreter,
  `numpy`/`polars`/`scipy` versions, the BLAS/LAPACK build NumPy was compiled
  against (`np.__config__.CONFIG`), and the loaded BLAS threadpools
  (`threadpoolctl`, when installed — `[]` on Accelerate/macOS).
  `fingerprint_sha256` digests the whole block so a determinism sweep
  compares one hash across machines instead of diffing nested config.
- `live_pnl_claim` is a `Literal[False]` — a receipt can never carry a live
  P&L claim. If `payload` echoes `data_label` or `live_pnl_claim`, the
  verifier requires agreement with the envelope, so an honest re-seal cannot
  launder a forged label.
- Forbidden headline metric keys (sharpe/sortino/calmar/pnl/nav) are scanned
  out of `payload`, with the same `live_pnl_claim` exemption the writers use.

## Writers

`build_receipt_v2(kind, data_label, dataset, params, code_files, verdict,
payload)` computes the digests and validates the envelope before returning
it — a malformed envelope raises instead of sealing. `seal_receipt` stamps
`receipt_sha256` over the canonical body; writers persist it atomically
(`fleet_eval._atomic_write_text` convention).

The first migrated lane is `dipcatcher fleet`:

```bash
uv run dipcatcher fleet --receipt-version 2   # default remains 1 (fleet_eval.v1)
```

`write_fleet_receipt(receipt, dir, receipt_version=2)` wraps the v1 payload
verbatim under `payload`; the v1 write path is unchanged.

## Verifier

```bash
uv run dipcatcher verify-receipt receipts/fleet_eval_<hash>.json
```

Prints `{valid, path, schema, kind, verdict, digest_convention, errors}` and
exits non-zero on any violation — fail closed on unreadable files, non-object
JSON, and unsealed receipts.

For `receipt.v2`: pydantic structure, the `receipt_sha256` seal, the
environment fingerprint, the `code_files` → `code_sha256` digest, the
forbidden-metric scan, and — for `distribution_fleet_eval` — re-derivation
of `dataset_hash`/`params_hash` from the embedded payload plus the
`verdict` ↔ `n_error_rows` agreement.

For older receipts: the `receipt_sha256` seal is checked under both repo
conventions — `canonical_json_bytes` (`digest_convention: "canonical_json"`)
and the strict `json.dumps` form used by `real_benchmark`
(`"strict_json"`); `fleet_eval.v1` payloads additionally get their writer
contract. Verification is integrity-only — it is not a digital signature
and never authorizes live trading (see `docs/RECEIPT_VERIFICATION.md` for
the Phase-1 notebook/run verifier).
