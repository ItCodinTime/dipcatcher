# Demo fixture — labeled SYNTHETIC

Tiny canonical slice extracted from `scripts/gen_demo_data.py` output
(`make demo-data`). Every row carries `source="synthetic"` /
`revision_id="SYNTHETIC"` — correctness fixture only, never market evidence.

- `bars.csv` — 6 daily bars (SEC_MKT + SEC_0001, first 3 sessions) in the
  bronze bar contract; exercises the `file` provider's CSV path.
- `security_master.parquet` — matching 2 security-master rows (kept as
  parquet so the all-null `valid_to` column keeps a temporal dtype; CSV
  inference would type it `String` and fail closed).
