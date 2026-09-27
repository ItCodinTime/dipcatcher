# PROOFCORE migration plan (additive coexistence)

Per DESIGN.md §13. Every phase is additive: no existing public API, receipt
format, or CI gate is weakened or removed at any point.

## Phase 0 — contracts (zero behavior change)

`quant_fund.proofcore.contracts` lands first. Existing suite stays green; the
module imports only stdlib + pydantic so all workstreams build against it
without import cycles. The one sanctioned existing-test edit in the whole
wave: the vacuous `test_live_requires_flag` (A3 #1) is replaced with a real
`pytest.raises(ValidationError)` guard.

## Phase 1 — PIT vault alongside Lake (W1)

`PitVault` + the migration shim land next to `quant_fund.data.lake.Lake`,
which stays untouched. The 32 direct `pl.read_parquet` sites keep working;
LH009 reports them as warnings. The vault reuses `require_pit_columns`-style
checks on append.

## Phase 2 — proof bundle primitives (W2)

W2 supplies fingerprinting, read recording, bundle construction, signing, and
independent bundle hash/sidecar/metric checks. The `quant proof run` and replay
entry points fail closed: a far-future whole-panel vault read cannot prove
what was known at each historical decision. An explicit decision schedule and
corresponding as-of reads are required before enabling them. `run_backtest`
and the receipt format are unchanged. Committed `receipts/*.json` remain valid
under their existing contracts. The heterogeneous committed receipt classes
do not yet have one universal verifier; `make receipts-reverify` fails closed
and is not a blocking CI gate. Future proof bundles live in `proofs/`
(gitignored, like `data/`); the provenance DB lives at
`data/metadata/proofcore.duckdb` (gitignored).

## Phase 3 — leakage hunter (W3)

`quant leakage scan` runs in CI in **warn mode** this wave (adjudicated):
report archived as the `leakage-report` artifact, findings advisory. The
seeded-leak fixture suite is blocking once present. The gate flips to
`--fail-on error` after one release of soak; LH009 (direct parquet reads)
flips to error only in the follow-up call-site migration wave.

## Phase 4 — reality filter (W4)

Unit-safe PSR/MinTRL, DSR/PBO/SPA/BH-FDR land, plus the scoreboard A1 F1 fix
(diagnostic values deflate by design — CHANGELOG-flagged). The CI
`reality-filter` job scores the provenance trial ledger with the unchanged
filter. `data/metadata/proofcore.duckdb` is not in the tree at HEAD or on
main (`data/metadata/**` is gitignored; only tests call
`ProvenanceDB.insert_trial`). Export of that missing path creates an empty
database and writes 0 rows. `insufficient_evidence` is a scored verdict and
needs rows, so `quant reality preflight --db` exits 3 with
`REALITY_FILTER_SKIP` before that empty database is created. `make
reality-gate` maps exit 3 to exit 0 and the workflow emits a notice. The
job fails when a recorded ledger's verdict is anything other than `pass`.

## Phase 5 — integration & CI gates on (W5)

- `.github/workflows/proofcore.yml` gate matrix active: proof-integrity (signer,
  recorder, and bundle verifier tests),
  leakage-scan (warn), reality-filter (blocking; empty ledger is a noticed skip),
  layering, coverage-floors, fx1-coverage.
  `docs/proofcore/proofcore.yml` retains a reference copy;
  `test_proofcore_ci.py` asserts the active body stays identical.
- Per-package coverage floors (`pit`/`proof`/`reality`/`proofcore` ≥ 90,
  `leakage` ≥ 85) enforced via `[tool.proofcore.coverage-floors]` in
  pyproject.toml — ADDITIVE to the existing global 80% floor, which is not
  lowered or otherwise touched. fx1 gets its first coverage gate
  (`--cov=fx1 --cov-fail-under=60`, ratchet) in the new workflow rather than
  by editing `fx1.yml`.
- Hypothesis runs under the derandomized `ci` profile
  (`HYPOTHESIS_PROFILE=ci`, DESIGN.md §9.5).
- External proof-chain head publication remains pending until persistent
  causally proven runs and replay verification exist.

## What is deliberately NOT in this wave

SCC decomposition of the 21 legacy packages, migration of the 32 rogue
parquet reads, ed25519 signing, changes to `research/verify.py` semantics,
fx1 code changes, and engine-semantics fixes (DESIGN.md §11). Each is gated
or flagged so the follow-up wave cannot ship silently.
