# Research-lane correctness audit — `src/quant_fund/research/`

Scope: all 29 top-level modules of `src/quant_fund/research/` (~14k LOC),
audited against the repo contract — fail-closed causality and honesty.
Look-ahead, silent NaN propagation, fabrications, forged/mutable receipts,
and non-contract error handling are defects. Protocol order: (a) receipt
writers and verifiers, (b) signal→return causality (walk-forward cursors,
per-date cross-sections, expanding windows), (c) forced SYNTHETIC stamps and
`FORBIDDEN_RESEARCH_METRIC_KEYS` gating, (d) fail-closed empty/degenerate/
missing-input handling. Verdicts: `CLEAN`, `FIXED` (defect repaired, with a
regression test), `DEFERRED` (real finding, not repaired here — blast radius
exceeds a minimal contract-preserving fix).

Regression tests for the fixes live in
`tests/unit/research/test_research_audit.py`.

## Bugs fixed

1. `benches/ranking.py::oos_rank_scores` — **FIXED**. When the panel has too
   few dates for the configured validation scheme, `walk_forward` returns no
   folds and the function fell back to a plain chronological holdout —
   `times[:cut]` fit, `times[cut:]` scored — with no boundary purge or
   embargo. The last `horizon_bars` train decisions carry labels that reach
   inside the holdout, so the "out-of-sample" IC/decile statistics were
   contaminated by realized test-period returns — the exact leak the fold
   path purges via `purge_mask` + `embargo_bars`. The fallback now applies
   the identical semantics (`purge_mask` against `[times[cut], times[-1]]`,
   then `idx[t] + embargo_bars < cut`); when purging leaves no train dates it
   fails closed to all-NaN scores instead of fitting on overlapping labels.
   Regression: `test_oos_rank_scores_small_panel_fallback_still_purges`,
   `test_oos_rank_scores_fallback_fails_closed_when_purge_empties_train`.
2. `sota_protocol.py::run_sota_protocol` — **FIXED**. `sota_receipt.json`
   was written by a bare `dest.write_text(json.dumps(receipt, default=str))`:
   non-atomic (a crash mid-write leaves a partial receipt on disk), unsealed
   (no `receipt_sha256` — `verify_receipt_file` reported
   `receipt_sha256_missing_or_invalid`), and type-corrupting (`default=str`
   silently stringifies foreign objects, changing the sealed content). The
   receipt is now canonicalized, self-sealed with `receipt_sha256` under the
   `canonical_json_bytes` convention, and published via
   `NamedTemporaryFile` + fsync + `os.replace` (the `agent.py`/"latest file"
   atomic-publish pattern). `verify_receipt_file(sota_receipt.json)` now
   passes with zero errors. Regression:
   `test_sota_receipt_is_sealed_and_self_verifying`.

## Module table

| module | verdict | evidence |
|---|---|---|
| `__init__.py` | CLEAN | Lazy `__getattr__` re-export only; no state or statistics; lazy-import guard covered by `test_catalog_lazy_import_guards` |
| `agent.py` | CLEAN | Hypothesis minting gated on finite p-values; `_atomic_write_text` (NamedTemporaryFile + fsync + `os.replace`) for overwritten artifacts; `immutable_json_sha256` binds digest-named receipts; receipts verify under `verify_receipt_payload` |
| `benches_extra.py` | CLEAN | Descriptive diagnostics on seeded SYNTHETIC streams; no promotion path |
| `benches_w810.py` | CLEAN | EnbPI readings on seeded SYNTHETIC AR(1) streams; honest scope stamps |
| `capacity_overlay.py` | CLEAN | Delay-1 causal capacity scales; seeded synthetic books; no caller-supplied stamps |
| `cost_allocation.py` | CLEAN | cvxpy solution re-verified by an independent check before use; degenerate inputs fail closed |
| `cross_sectional.py` | CLEAN | Seeded panel + planted signal; output stamped SYNTHETIC correctness evidence |
| `fleet_eval.py` | CLEAN | `inputs_sha256` binds the scored shard set; digest-named receipts published via `os.link` (refuse overwrite — immutable); `fleet_v1_contract_errors`/`fleet_v2_consistency_errors` re-derive claimed stats; one-step-only forecasts scored |
| `forward_evidence.py` | CLEAN | Bartlett HAC on forward shadow deltas; degenerate series → inconclusive |
| `forward_shadow.py` | CLEAN | Shadow journal is append-only evidence; no mutable claimed statistics |
| `garch_benchmark.py` | CLEAN | `har_forecast` is causal; chronological val/test separation preserved |
| `identity_sweep.py` | CLEAN | Estimator errors become NaN+error rows, not fabricated residuals; `data_source == "SYNTHETIC"` asserted before publish; self-hash seal; atomic write |
| `net_replay.py` | CLEAN | Replay statistics derived from recorded tape; no caller stamps |
| `net_tournament.py` | CLEAN | Manifest sealed before scoring; attempt-reservation prevents score reuse |
| `phase1_verify.py` | CLEAN | Phase-1 artifact verification; fail-closed on missing/unreadable evidence |
| `prospective_sota.py` | CLEAN | Prospective journal semantics; claims gated on sealed evidence |
| `ranker_probability.py` | CLEAN | Purged + embargoed walk-forward throughout; Platt scaling fit on calibration split only; `prediction_sha256` binds predictions |
| `real_benchmark.py` | CLEAN | Synthetic inputs rejected outright; real-data path keeps market/synthetic separation |
| `reality_sweep.py` | CLEAN | Pre-registered sweep; holdout excluded from PBO computation; Ed25519 attestation |
| `receipt_schema.py` | CLEAN | Schema constants + v1→v2 migration shim; no mutable state |
| `receipt_v2.py` | CLEAN | `verify_receipt_payload` routes `schema == "receipt.v2"` **or** `schema_version == 2` to the strict v2 path (pydantic envelope + canonical/strict seal + environment/code/kind re-derivation) — it can over-reject, never under-validate; v1 checks seal + `live_pnl_claim` + fleet_eval contract |
| `research100.py` | CLEAN | Thin catalog runner; no statistics of its own |
| `research100_benchmark.py` | CLEAN | Output forced SYNTHETIC, `promotable` False |
| `research100_cli.py` | CLEAN | `--synthetic` flag required; cannot emit unlabeled output |
| `shadow_journal.py` | CLEAN | SQLite hash chain + immutable triggers; journal tamper-evident |
| `sota_evidence.py` | CLEAN | Evidence assembly re-derives claims from receipts; no mutable writers |
| `sota_protocol.py` | FIXED | Unsealed, non-atomic, `default=str`-corrupting `sota_receipt.json` write → canonical seal + atomic publish (see above); frozen `SotaProtocol` (extra=forbid), `protocol_sha256`, as-of causality (`event_time <= asof` + `available_time <= asof`), calibration gate, and `promotion_decision` (SYNTHETIC can never promote, `blend_weight` stays 0) all verified clean |
| `verify.py` | CLEAN | `verify_research_artifact` dispatch; fail-closed on unreadable/foreign payloads |
| `vol_bench.py` | CLEAN | Seeded vol shards; receipts under `receipts/` with the `os.link` immutable convention |

## Support lanes (read for context — not part of the 29-module count)

| lane | verdict | evidence |
|---|---|---|
| `benches/ranking.py` | FIXED | `oos_rank_scores` no-fold fallback (fix #1); `_policy_ridge_topk_rewards` uses a causal expanding prefix; pairwise DM aligns on common date keys only |
| `benches/common.py`, `benches/families.py`, `benches/intervals.py` | DEFERRED | `_holdout`/`_triple_split`/`_gaussian_interval_split` are plain chronological cuts with no boundary label purge — every bench arm shares the identical tr/cal/te cut so comparisons stay honest, but the boundary train rows' labels overlap the holdout. Purging needs the label horizon threaded through ~15 call sites — a convention change wider than a minimal fix; flagged for a follow-up |
| `catalog/` (registry, dispatch, receipt helpers, shims) | CLEAN | `FORBIDDEN_RESEARCH_METRIC_KEYS` gates output blobs (`family_blob_forbidden_metrics_absent` scans keys, case-insensitive); ~150 soft-verify helpers uniformly NaN-skip / range-fail; re-export shims hold no logic |
| `explainability/` | CLEAN | Proper-score adapters only; sidecar pattern never mutates the sealed receipt; `relative_to` traversal guard; seeded permutation importance reports honest negatives |
| `metrics/agreement.py`, `metrics/concordance.py`, `metrics/inference.py` (protocol-named stats) | CLEAN | kappa/gamma/tau-b/c-index and DM/HAC/clustered-t all filter non-finite and return inconclusive/NaN on degenerate input — fail-closed, never fabricated |
| `utils/hashing.py` | CLEAN | `canonical_json_bytes`/`receipt_tree` canonicalize numpy scalars/arrays, non-finite→None, datetime→isoformat, bytes→hex, sets→sorted — the convention every seal above relies on |
