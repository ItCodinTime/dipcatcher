# PIT vault enforcement audit — `src/quant_fund/pit/` + `proofcore` manifest contracts

Scope: `pit/vault.py`, `pit/manifest.py`, `pit/corrections.py`,
`pit/frame.py`, `pit/shim.py`, `pit/cli.py`,
`proofcore/contracts.py` (`PitManifest`, `PitManifestFile`, chain anchors),
`data/point_in_time.py` (legacy filter the vault supersedes), and the
`leakage/ast_scan.py` LH009 gate. Every file was read end-to-end; candidate
weaknesses were probed live against a real `PitVault` before being listed
here. Regression KATs live in `tests/unit/pit/test_vault_enforcement.py`
(22 tests, deterministic, offline).

## Threat model verified

- `asof(t)` is the only read path: `known_at <= t` is applied inside the scan
  before caller code sees the frame (`corrections.select_asof`), re-checked
  by `PitFrame.validate` before return.
- Manifest chain: `manifests/rNNNNNNN.json` retained revisions, the
  `manifest.json` pointer must byte-equal the last retained revision, and
  `manifest.sha256` sidecar commits to the previous revision's sha256
  (GENESIS_HASH at creation).
- Write-once parts: `parts/r{revision:07d}.parquet` published by
  `NamedTemporaryFile` + `os.link`, which cannot replace an existing path.
- Every committed part is re-hashed on read (`_verified_parts`) before
  polars ever scans it.

## Bugs fixed

1. `dataset.json` unanchored (`vault.py`, `manifest.py`, `contracts.py`) —
   **FIXED**. The per-dataset metadata file (`security_level`,
   `monotonic_known_at`) was written at `create_dataset` and never committed
   into the manifest chain, so a one-bit tamper (`security_level: true →
   false`) silently collapsed asof output (probe: 2 rows → 1 row by dropping
   the security_id leg of the group-by). `PitManifest` now carries
   `dataset_meta_sha256` — sha256 of the dataset.json bytes, committed at
   creation and re-verified on every `read_manifest`. Legacy manifests parse
   with `GENESIS_HASH` (unanchored grandfathering; documented, not silently
   "fixed" since rewriting retained revisions is itself a chain violation).
   `append` inherits the anchor; `recover_interrupted_manifest` requires the
   pending manifest to carry the same anchor.
2. `dataset.json` name field unchecked (`vault._dataset_meta`) — **FIXED**.
   A foreign dataset's `dataset.json` copied over a victim's (swapped whole
   file) parsed fine and silently supplied the foreign `security_level`/
   watermark policy. `_dataset_meta` now requires `meta["dataset"] == name`.
3. Manifest structural shape unchecked (`manifest._validated_history`) —
   **FIXED**. A forged revision-0 manifest listing a part file passed every
   hash check (there is no "must have zero files" rule at rev 0) and died
   only later at part-hash verification — a consistent forgery could skip
   revisions or misname entries. New `_check_manifest_shape`: `revision ==
   len(files)` and `files[i].path == {dataset}/parts/r{i+1:07d}.parquet`,
   enforced on every retained revision in the chain and on pending manifests
   in `recover_interrupted_manifest`.
4. Metadata anchor could drift mid-chain (`manifest._validated_history`) —
   **FIXED**. A retained mid-chain revision rewritten with a different
   `dataset_meta_sha256` is rejected with "metadata anchor drift" before the
   broken successor link is even evaluated.
5. `recover_uncommitted_part` symlink-order bug (`vault.py`) — **FIXED**.
   `if not orphan.exists(): return False` ran before the `is_symlink()`
   check, so a **dangling** symlink at `parts/rNNNNNNN.parquet` reported
   `exists() == False` → recover returned False while the link still wedged
   every subsequent append's write-once check (`os.link` refuses). The
   symlink check now runs first; dangling links fail loud as "unsafe
   uncommitted part" for operator removal.
6. Nested dataset dirs (`vault.create_dataset`) — **FIXED**.
   `create_dataset("silver/bars/parts")`-style names placed a child's
   `parts/` machinery inside the parent's namespace: the child's first
   append landed `parts/r0000002.parquet` as a *directory* under the parent,
   permanently wedging parent appends and polluting `verify()` glob scans.
   Names that nest under or over an existing dataset are now rejected.
7. Duplicate `(key, known_at)` rows inside one append
   (`corrections.require_pit_frame`) — **FIXED**. Two rows claiming to be
   versions of the same security at the same publication instant were
   silently resolved by arrival order (last wins) in `select_asof`. Ambiguous
   versions now fail closed.
8. `_dataset_write_lock` leaked `sqlite3.OperationalError` (`vault.py`) —
   **FIXED**. A busy/corrupt append lock surfaced as raw `sqlite3.Error`,
   escaping the `VaultError` taxonomy callers catch. Now wrapped.
9. `pit stats` crashed on a corrupt dataset (`pit/cli.py`) — **FIXED**.
   `read_manifest` `ManifestError` propagated and killed the whole listing;
   stats now prints `manifest unreadable` for that dataset and continues.
10. Shim accepted naive `event_time` (`pit/shim.py`) — **FIXED**.
    `_to_pit_frame` enforced tz-aware `known_at` but not `event_time`; a
    legacy frame with a naive event-time column passed the gate.
11. LH009 did not gate `PitVault.history()` (`leakage/ast_scan.py`) —
    **FIXED**. `history()` is the unfiltered all-versions audit API — its
    own docstring claims LH009 protection, but the AST rule only flagged
    `read_parquet`/`scan_parquet`/`getattr`/sql reads. `x.history(...)` calls
    now emit LH009 (warning severity, matching the adjudicated posture).
12. Dead conditional (`corrections.select_asof`) — **FIXED**. `sort_keys`
    had identical branches (`list(key_cols) if columns is None else
    [c for c in key_cols]`); collapsed to `list(key_cols)`.

## Invariants pinned by tests (pre-existing, verified correct)

- Out-of-order / forged `prev_manifest_sha256` in any retained revision →
  `ManifestError` ("manifest chain broken at revision N").
- Forged per-file `sha256` in a manifest entry → `ManifestError` on read and
  a `verify()` violation.
- `asof(t)` boundary inclusivity: `known_at == t` is returned; a dataset
  whose only rows carry `known_at > t` → `VaultUnavailableError`.
- Symlinked part path escaping the dataset → `part_path` rejects it; a
  dataset dir symlinked outside the vault root → `VaultError` before any
  file is read.
- `PitManifestFile.validate_path`: vault-relative `parts/rNNNNNNN.parquet`
  only — no absolute paths, `..`, backslashes, or non-canonical names.
- `write_manifest` refuses to overwrite an existing retained revision and
  refuses a `prev_manifest_sha256` that disagrees with the sidecar anchor.

## Design limits (documented, not fixed)

- **Full-chain rewrite is undetectable.** A forger who rewrites *all*
  retained revisions plus the sidecar produces a self-consistent chain with
  different bytes — sha256 chains anchor history to itself, not to an
  external root of trust. Mitigations would need an external anchor
  (signed head manifest, transparency log); out of scope for this lane.
- **`PitFrame` is shallow-immutable.** The frozen dataclass wraps a mutable
  `pl.DataFrame`; a caller can mutate `frame.frame` in place. Rows are only
  guaranteed at return time.
- **`history()` remains reachable in-process.** LH009 is a warning-severity
  lint gate, not a runtime interposer; a strategy that ignores the lint can
  still call it. Considered acceptable under the adjudicated LH009 posture.
- **Unanchored legacy manifests** (pre-`dataset_meta_sha256` vaults) keep
  reading but do not detect dataset.json tampering — by design for backward
  compat; `create_dataset` always anchors new datasets.

## Interaction with `data/point_in_time.py`

The legacy helpers (`filter_available`, `validate_feature_frame`,
`filter_trailing_returns_asof`) filter `available_time` for frames that have
not been through the vault (lake reads mapped by `pit/shim.py`, which aliases
`available_time → known_at`). One inconsistency noted:
`filter_trailing_returns_asof` fails closed on null `available_time` only
when a `ret_1` column is present and non-null — frames lacking `ret_1`
silently drop null-availability rows through the filter, where
`validate_feature_frame` would have rejected them. Left as-is: tightening it
would change covariance behavior outside this lane's scope; flagged for the
data lane.
