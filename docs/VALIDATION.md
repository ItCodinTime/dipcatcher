# Validation

## Forbidden

Random shuffled train/test splits for primary financial experiments.

## Walk-forward

**Expanding:** train \([t_0, t_k]\), validate \((t_k, t_{k+1}]\), test \((t_{k+1}, t_{k+2}]\), then grow train.

**Rolling:** fixed-length train window.

Folds are built from **decision timestamps**, not shuffled rows.

## Purging

If a training label's observation window overlaps a validation/test decision period, drop that training row. Overlap uses label horizon \(h\) in trading sessions.

## Embargo

After a validation block, drop the next \(e\) sessions from training of the subsequent fold. Default \(e = h\) (the label horizon). Config: `validation.embargo_bars`.

## CPCV

Combinatorial purged cross-validation: split the timeline into \(N\) sequential groups, form all combinations of \(k\) groups as test, train on the complement with purge+embargo. Used as robustness, not as the only selection criterion.

**Per-group purge (Wave 25 correctness):** purge and embargo are applied
**per contiguous test group**, not across the union span of all test groups
in a combinatorial fold. A union-span purge would incorrectly wipe intervening
train dates that sit between non-adjacent test blocks. Embargo still drops
the configured bars immediately before/after each test block. Empty purged
trains are skipped (fail-closed). See `validation/cpcv.py` and
`tests/unit/test_cpcv_extremes.py`.

The research notebook includes a `cpcv` integrity family that records the
observed versus expected fold count and validates date-level train/test
separation. It is a validation audit, not a performance or profitability
claim.

## Hyperparameters

Optuna may see **train** (and inner CV) only. Model selection uses validation. The final test split is untouched until evaluation. Trial count is stored for DSR.

## Promotion levels

Promotion is fail-closed. A candidate must provide explicit finite evidence for
`mean_ic`, `net_spread`, and `turnover`, set `evidence_complete=true`, and pass
the leakage gate, and be bound to a verified immutable research receipt.
Missing metrics or a missing/invalid receipt are not treated as zero or as
optional metadata. `SYNTHETIC` data is
useful for correctness and benchmark recovery, but can never receive a
production promotion alias.

The registry also enforces the boundary: `candidate`, `shadow`, and `retired`
are workflow aliases, while `champion` requires an approved promotion receipt
with complete evidence and a non-empty run identity.

A champion requires all configured levels:

1. Better than trivial statistical baseline
2. Better than econometric baseline
3. Stable across walk-forward folds
4. Economic value before costs
5. Economic value after costs
6. Reasonable cost-sensitivity (does not vanish at \(2\times\) impact)
7. Risk limits respected
8. No leakage-test failures; DSR/PBO warnings visible

Failure at any level remains in the report. Improving RMSE while net P&amp;L worsens does not promote.

`cost_sensitivity` runs matched configured and doubled-impact scenarios through
the same event-driven engine and reports costs, returns, and rejects as
`execution_diagnostic_only`. It is not a live-performance claim.

## Leakage suite

Injected future price, future fundamental, future CS statistic, future universe membership must cause tests to fail (assert the detector raises).

## Production monitoring safety

Drift reports expose PSI, mean shift, sample counts, and an explicit
`insufficient_data` state; short windows are never reported as healthy. Any
PSI or configured mean-shift breach is an alert. The kill switch blocks new
orders in halt/cancel/flatten states, and flattening always requires human
authorization even when an auto-flatten flag is present.

The pre-trade risk gate also rejects non-finite inputs, non-positive NAV or
prices, zero quantities, stale prices, and stale model outputs before applying
notional, concentration, gross, net, participation, and volatility limits.

Conformal prediction sets also enforce ordered lower/upper bounds after
expansion, so malformed upstream intervals cannot become negative-width
research diagnostics.

## Synthetic markets

E2E recovery of a planted factor is labeled `SYNTHETIC`. It tests the engine, not profitability.

## CLI: `dipcatcher validate`

Fail-closed gates implemented in `quant_fund.validation.gates.validate_candidate`:

1. Require a structurally valid causal target-weight panel (non-null keys,
   finite weights, unique as-of/security rows) **or** walk-forward /
   research-notebook evidence.
2. Reject SYNTHETIC when `--claim-live` is set.
3. Require `metadata/research/latest.json` to pass the immutable research-receipt verifier before promotion.
4. Require the candidate `run_id` to exactly match the receipt provenance `run_id`; missing or mismatched identities fail closed.
5. Delegate promotion to `promotion_decision` (missing metrics are not zeros; SYNTHETIC never promotes).
6. When a research notebook exists, expect BH-FDR family tags (`calibration` / `discovery` / `bound`).

Exit code 0 only when research-correctness `ok` is true. `promote` is a separate boolean.

## Overnight status (Wave 15 — 2026-09-15/16 IST)

Honest research/validation posture after Waves 1–15 (no live P&L claims):

- **Multi-fold stability** — `promotion.min_folds` (default 2) and optional
  `fold_ic_stability` are enforced in `promotion_decision` and
  `validate_candidate` (`multi_fold_stability` gate). Insufficient folds fail
  closed for both research-ok and promote.
- **Diebold–Mariano** — pairwise ranking −IC DM is recorded in the research
  notebook / CLI (oracle vs ridge families). DM is a forecast-accuracy audit,
  not a profitability claim.
- **BH-FDR families** — calibration / discovery / bound are split and never
  pooled in the research CLI banner (`format_fdr_families`).
- **`synthetic_not_promotable`** — `data_source=SYNTHETIC`, `synthetic=true`, or
  config `data.source=synthetic` always yields `promote=false` via
  `synthetic_evidence_not_promotable`. Perfect IC/spread/turnover cannot override.
- **`claim_live` fail-closed** — `--claim-live` on SYNTHETIC sets `ok=false` with
  `synthetic_claimed_as_live` (in addition to never promoting).
- **Paper promotion dry-run honesty** — paper/shadow ledger
  `promotion_dry_run.would_promote_live` is always `false`;
  `live_pnl_claim=false`, `research_only=true`. Schema validators reject any
  receipt that claims otherwise.
- **Online CRC extremes** — empty/short calibrate, bad `B`/γ/α, λ≥0 floor, and
  miss-streak honesty covered in `tests/unit/test_online_crc.py` (risk keys only;
  no Sharpe-as-live).

Wave 15 SYNTHETIC smoke (research-only — **NOT** live P&L): `dipcatcher research`
oracle_raw IC≈0.79; `quant validate` ok=true promote=false; `--claim-live`
ok=false; `dipcatcher paper --max-steps 8` → 173 fills, would_promote_live=false.
