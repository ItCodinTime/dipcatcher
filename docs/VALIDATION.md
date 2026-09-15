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

## Hyperparameters

Optuna may see **train** (and inner CV) only. Model selection uses validation. The final test split is untouched until evaluation. Trial count is stored for DSR.

## Promotion levels

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

## Leakage suite

Injected future price, future fundamental, future CS statistic, future universe membership must cause tests to fail (assert the detector raises).

## Synthetic markets

E2E recovery of a planted factor is labeled `SYNTHETIC`. It tests the engine, not profitability.
