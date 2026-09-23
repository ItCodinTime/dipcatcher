# Top-10 execution plan

This is an evidence-ranked implementation queue for the current Dipcatcher
worktree. Priority reflects research value, safety impact, and whether the
repository already contains enough contracts to implement the task without
inventing live-data evidence. Tasks are sequential; each closes with focused
tests, static checks, and an explicit evidence note.

## Queue

| Rank | Task | Why it is worth doing | Current evidence / acceptance gate |
|---:|---|---|---|
| 1 | Establish a fail-closed GARCH consumer contract | The GARCH implementation exists but is date-level pooled while forecast volatility is per-security; direct wiring would be scientifically invalid. | Scope validator must reject pooled artifacts at per-security consumers, use `forecast(...)['variance']` only, preserve units, and pass PIT/QLIKE tests without forward labels. |
| 2 | Add a typed model-artifact manifest and provenance binding | Checksum sidecars detect corruption but do not authenticate origin; model cards require git, feature, label, universe, and dataset identity. | **Partial:** atomic `model_artifact.v1` manifests bind payload hash, class, features, and payload-provided provenance; training reports now add canonical config, git revision, dirty-worktree, and canonical materialized-panel hashes when available. Dataset identity still is not injected into every artifact manifest. |
| 3 | Make distribution and volatility model selection evidence-based | Ranking/RL/calibration have auto-selection, while distribution/volatility still require manual selection. | **Implemented with limits:** `auto` routes select the lowest finite causal walk-forward mean pinball/QLIKE only when candidate OOS rows meet configurable `auto_min_oos_rows`; selected aliases, NaN exclusion, and undersized-sample rejection are tested. Promotion policy remains separate. |
| 4 | Define and implement an opt-in runtime calibration contract | Calibration artifacts now exist, but forecast confidence must not consume a score with different semantics. | **Implemented with explicit opt-in:** runtime loading fails closed unless the fitted artifact is present, bound to score/label/horizon identities, includes fit/OOS bounds, and satisfies optional as-of freshness limits. |
| 5 | Complete per-security volatility scope or keep pooled GARCH isolated | Current GARCH is date-level equal-weight pooled volatility, not an asset-specific forecast. | **Isolated:** pooled artifacts now carry `date_level_equal_weight_cross_section` scope and reject per-security consumers; a keyed per-security implementation remains future work. |
| 6 | Expand non-synthetic PIT data adapters and provenance | Readiness explicitly blocks live claims because only synthetic/local files exist. | **Existing boundary verified:** PIT/collector/public-source regression slice passes (40 tests); collectors require `event_time`, `available_time`, `ingested_time`, source, and revision, enforce the causal chain, and write hashed PIT receipts. External coverage expansion remains. |
| 7 | Add model-family health and drift monitoring | Monitoring exists, but model artifact health, feature drift, and calibration drift need one fail-closed operational report. | **Partial:** `model_health_report` covers artifact manifests, missing/insufficient features, PSI alerts, optional Brier calibration drift, and research-only output; `/monitoring/drift` now verifies the evidence-report hash sidecar before exposing persisted status. Broader family wiring remains. |
| 8 | Reduce causal forecast latency without parallelizing unsafe state | Performance docs identify conformal Student-t MLE as the dominant remaining cost. | **Evidence review:** bounded LRU wrappee caching, shared causal frames, and cache fingerprints already exist; current cache/reselect/causal suites pass (28 tests), and the checked-in synthetic benchmark reports ~29.5x cold-to-hot speedup. A fresh wall-clock rebenchmark remains; unsafe date parallelism remains rejected. |
| 9 | Harden promotion/registry from research artifact to paper challenger | Registry and gates exist, but new auto-selected/calibrator artifacts need uniform promotion receipts. | **Partial:** verified artifact identity is wired through training results/reports and MLflow tags; both promotion decision and champion approval now reject malformed/unverified artifact identity when supplied. Full promotion-receipt composition remains. |
| 10 | Build an institutional evidence report for every selected model | Current diagnostics are distributed across MLflow, receipts, and docs. | **Implemented:** `evidence_report.v1` assembly, atomic JSON/Markdown persistence plus JSON hash sidecar, automatic training-dispatcher emission, artifact path/hash/config identity, health/promotion sections, and fail-closed completeness warnings are tested. |

## Chosen order and constraints

Task 1 is first because it is local, testable, and already specified by the
GARCH model card. Vendor data and live broker work are deliberately later:
they require external authorization and cannot be honestly simulated into
readiness. No task may weaken PIT, purging, synthetic-data labels, or
research-only promotion gates to improve a metric.

## Working protocol

For each task: inspect current contracts; write or update tests first where the
behavior is missing; implement the smallest complete vertical slice; run the
focused tests plus Ruff/mypy/diff checks; then update this file with evidence,
remaining limitations, and the next task.

## Status

- [x] 1. GARCH consumer scope contract (pooled artifacts reject per-security use; 53 focused GARCH/training tests pass)
- [~] 2. Artifact manifest and provenance binding (config/git/panel provenance emitted in reports; universal artifact-manifest injection pending)
- [~] 3. Distribution/volatility auto-selection (finite-score and minimum-OOS policy implemented; promotion policy remains separate)
- [x] 4. Opt-in calibration runtime contract (identity, required window metadata, freshness policy, and branch tests implemented)
- [x] 5. Per-security volatility scope (pooled GARCH explicitly isolated; per-security model not claimed)
- [~] 6. Non-synthetic PIT adapters (PIT/provenance contract exists; adapter coverage expansion pending)
- [~] 7. Model health/drift monitoring (unified report and API exposure implemented; family-specific calibration drift pending)
- [~] 8. Causal performance optimization (existing safe cache optimizations verified; fresh rebenchmark/equivalence gate pending)
- [~] 9. Registry/promotion hardening (MLflow identity attachment implemented; full promotion-receipt composition pending)
- [x] 10. Institutional evidence report (schema, durable writer, automatic emission, artifact linkage, and fail-closed completeness gate implemented)
