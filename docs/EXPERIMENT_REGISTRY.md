# Experiment registry — MLflow as index, receipts as evidence (P7.3)

`mlflow.db` exists locally because the `train *` lanes log runs to it
(`registry/mlflow_store.py`). ULTRAPLAN P7.3 asked whether fleet tournament
runs should be wired into the same registry or documented why not. The
decision: **wire, but as an index — never as evidence.**

## The two artifact classes

| | Receipt (`receipts/*.json`) | MLflow run (`mlflow.db`) |
|---|---|---|
| Mutability | Immutable — `_atomic_write_text` refuses overwrite with different content | Mutable — tags and aliases change in place |
| Addressing | Content-addressed — sealed sha256 over canonical bytes | Server-assigned run_id, non-reproducible |
| Durability | Committed evidence (`receipts/` is tracked input) | Gitignored local sqlite — regenerable, per-machine |
| Role | The evidence; `verify-receipt` audits structure + digests | The index; discovery query surface ("which runs exist, what scored best") |

Two stores for the same result is a split-brain risk — so the index row does
not duplicate evidence, it *points at it*: every fleet run carries a
`receipt_sha256` tag over the sealed file bytes plus `evidence=receipt`.
Auditing an index row means re-verifying the receipt it names, not trusting
the row's metrics.

## What gets logged

`dipcatcher fleet --mlflow` (opt-in, default off) calls
`log_fleet_run(receipt_path, receipt)` after the receipt is sealed:

- **Params**: seed, n_train, n_eval, tau grid, models, shards, receipt
  schema, receipt path.
- **Metrics**: proper scores only — `mean_crps`, `mean_pit_ks`,
  `best_head_crps` / `worst_head_crps` (per-head mean CRPS extrema), plus
  `n_heads` / `n_shards` / `n_ok_rows` / `n_error_rows` counts. Only
  `status == "ok"` result rows aggregate; an all-error tournament fails
  closed instead of logging an empty claim.
- **Tags**: `data` (the receipt's own `data_label`), `kind`,
  `evidence=receipt`, `receipt_sha256`.

The experiment name is `fleet_tournament`, run name `fleet-<sha[:12]>`.

## Why not the alternatives

- **Receipts as the index**: receipts are deliberately write-once files —
  no query surface, no comparison view. Discovery is the thing mlflow is
  for.
- **MLflow as evidence**: run rows are mutable (tags move, metrics can be
  re-logged), live in a gitignored per-machine sqlite, and carry
  non-reproducible run ids — none of the three properties the honesty
  contract requires of evidence.
- **No wiring**: fleet tournaments would be invisible to the registry —
  the training lanes would have an index the tournament lane lacked.

## Promotion safety

`promotion_is_approved` already fails closed on `data_source == "SYNTHETIC"`
and `synthetic == true`, so a fleet index row can never satisfy the champion
alias gate — the wiring adds discoverability without a new promotion path.

See `RECEIPT_V2.md` for the sealed-envelope schema and
`RECEIPT_VERIFICATION.md` for the audit path.
