# legacy-unsealed/

Artifacts committed here predate the `receipt_sha256` envelope convention
(`receipt.v2` and the sealed `*.v1` writers). They are retained for
provenance — content is unchanged, git history intact — but **they cannot be
re-verified**: no top-level seal exists to check, and the schema-specific
honesty verifiers (`verify-research`) target the research-notebook schema,
not these bench-result formats.

Because of that, this directory is deliberately **outside** the
`receipts/*.json` glob used by `receipts-reverify` and `verify-all`: the
sealed set must be end-to-end re-verifiable, fail-closed. If any of these
experiments are re-run under a sealed writer, the new receipt belongs in
`receipts/` proper.
