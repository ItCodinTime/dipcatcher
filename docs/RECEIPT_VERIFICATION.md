# Phase-1 receipt verification

`dipcatcher verify-research` accepts three evidence surfaces:

```bash
uv run dipcatcher verify-research
uv run dipcatcher verify-research data/metadata/real_benchmark/us_wide_v1
uv run dipcatcher verify-research data/metadata/net_tournament/us_wide_v1
uv run dipcatcher verify-research data/metadata/research/phase1_evidence_index.json
```

The default and any ordinary JSON notebook path retain the canonical notebook
verifier. A run directory selects the Phase-1 verifier from its sealed
`manifest.json`. An index JSON has `kind: "phase1_evidence_index"`.
Verification returns JSON with `valid`, `kind`, `state`, and errors; the CLI
exits nonzero when invalid. A **verified blocked** tournament has `valid=true`
and `state="blocked"`: its validation attempt/report are intact and its
frozen slate failed or was untestable, so there is no test attempt, test
receipt, selected candidate, or economic success claim. It is a reviewable
failure record, not a passed tournament.

Large phase receipts may be published as `validation.json.gz` or
`test.json.gz`. The verifier decompresses them and checks the same JSON seal
and index link. It rejects a directory containing both compressed and raw
copies of one receipt. The tournament runner itself expects raw JSON when
loading a validation receipt, so restore a published archive with `gzip -dk`
before running an additional phase in a fresh workflow. Remove the compressed
copy from that working directory before verification, or the two representations
will be rejected as ambiguous.

The run verifier checks each canonical JSON SHA-256 seal, requires validation
and test receipts for a complete run, verifies the tournament's phase attempt
receipts and validation-to-test link, checks the embedded benchmark manifest
against the parent run on disk, hashes the source Parquet bytes, checks code
hashes and dependency versions against the current checkout, and checks
matched baseline/trial coverage and research-only flags. It does not reproduce
the forecasts or replay every order. It preserves failed trials and rejects a
forged promotion/live claim. A blocked tournament requires its sealed
validation and forbids either test file.

The optional evidence index binds the completed and blocked runs to frozen
config files and a declared Git/catalog context. Its paths are relative to
the index file, so an independent clone can relocate the whole tree together.
The index itself uses the same canonical JSON seal: SHA-256 of its fields
excluding `receipt_sha256`, encoded with sorted keys and compact separators.

```json
{
  "kind": "phase1_evidence_index",
  "schema_version": 1,
  "created_at": "2026-09-25T06:30:00+00:00",
  "git_revision": "<40-lowercase-hex-commit-id>",
  "git_worktree_sha256": "<64-lowercase-hex-digest>",
  "benchmark_catalog_version": 2,
  "runs": [
    {
      "kind": "real_benchmark",
      "path": "../real_benchmark/us_wide_v1",
      "config_path": "../../../configs/real_benchmark_us_wide.json",
      "config_sha256": "<sha256-of-config-file-bytes>",
      "manifest_sha256": "<manifest-receipt-sha256>",
      "validation_sha256": "<validation-receipt-sha256>",
      "test_sha256": "<test-receipt-sha256>"
    },
    {
      "kind": "net_tournament",
      "path": "../cost_aware_tournament/us_wide_v1",
      "config_path": "../../../configs/cost_aware_tournament.json",
      "config_sha256": "<sha256-of-config-file-bytes>",
      "manifest_sha256": "<manifest-receipt-sha256>",
      "validation_sha256": "<validation-receipt-sha256>",
      "test_sha256": null
    }
  ],
  "receipt_sha256": "<sha256-of-all-other-fields>"
}
```

`test_sha256: null` is valid only when the linked tournament is verified
blocked, with no test receipt or attempt. The verifier checks each config's
raw-byte hash and agreement with the frozen protocol or slate. The index's
Git revision must identify a local commit whose source blobs match the run
code hashes. A run made from a dirty checkout can instead match the current
revision and the repository's current dirty-worktree fingerprint; this cannot
be independently reconstructed after those changes disappear. The index is
not a digital signature or an independent attestation of past worktree
contents. Source
entitlements, vendor adjustments, historical availability reconstruction,
survivorship, undisclosed experiments, and prior holdout inspection remain
disclosed limitations. Passing verification never authorizes live trading.

Legacy manifests containing an absolute dataset path can be verified where
that original path exists; they are not portable to a new checkout. New
manifests store the dataset relative to the benchmark run directory, and
tournament manifests store the parent benchmark path relative to their own
run directory.
