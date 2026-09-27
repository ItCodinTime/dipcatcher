# Backtest overfitting diagnostics

Every Dipcatcher research run stamps five diagnostics on the notebook
(`backtest_overfitting`) and prints them in the markdown report. They
describe the selection procedure. They are not a performance claim, and
they are not an authorization to trade.

The per-period input is the date-level score the runner already computes
for each ranker (the information coefficient). Bailey & López de Prado's
formulas take the ratio of that series' mean to its standard deviation,
in the sampling frequency of the series. The receipt stores `pbo`, `dsr`,
`psr`, `min_trl`, `n_trials`, and `n_trials_effective`. It does not store
a headline portfolio ratio.

## How to read each number

| Field | Range | Reading |
| --- | --- | --- |
| `pbo` | 0 to 1 | Fraction of combinatorially symmetric splits in which the in-sample winner is below the out-of-sample median. Near 0, the winner tends to stay above the median out of sample. Near 1/2, the winner is a coin flip (the pure-noise rate in Bailey, Borwein, López de Prado & Zhu 2017). Above 1/2, the in-sample winner is below the median more often than not. |
| `psr` | 0 to 1 | Probability that the selected series' true mean/std exceeds 0, after skewness and raw kurtosis (Bailey & López de Prado 2012). A value above 0.95 is the paper's usual confidence bar for a single trial. |
| `dsr` | 0 to 1 | The same probability with the hurdle raised to the expected maximum across the effective number of independent trials (Bailey & López de Prado 2014). It is at most `psr`. Below 0.95, the selected score does not clear a 95% bar once the search is counted. |
| `dsr_counted_trials` | 0 to 1 | The same deflation counting every evaluated config, with no clustering reduction. |
| `min_trl` | ≥ 1, in observations | How long the selected series must be before `psr` against 0 can reach 95%, holding the observed moments fixed (Bailey & López de Prado 2012). Compare it with `n_obs`. |
| `n_trials` | integer ≥ 0 | Configs the runner evaluated, including ones that could not be aligned onto a common date grid. |
| `n_trials_effective` | integer, ≤ `n_trials` | Independent-trial count after clustering. |

`metrics_status` is `computed` when a finite diagnostic was produced,
`unavailable` when the run had too little aligned history, and
`legacy_uncomputed` on a schema-1 receipt that was migrated without
recomputing anything.

## Combinatorial purged cross-validation

López de Prado (*Advances in Financial Machine Learning*, 2018, ch. 7 and 12)
splits the timeline into `N` contiguous groups and forms every combination
of `k` groups as the test set: `C(N, k)` splits. Training rows whose label
window reaches a test block are purged. An embargo drops a fixed number of
bars on each side of every test block. Purge and embargo run per test
block, so a train block sitting between two test blocks is kept when its
label does not reach either block.

The `C(N-1, k-1)` backtest paths stitch those splits so that each group is
tested once per path and no test forecast is reused. The receipt records
`cpcv_n_splits`, `cpcv_n_paths`, `cpcv_n_folds` (folds that still have a
train set after purging), and `purge_disjoint`.

## Probability of backtest overfitting

CSCV (Bailey et al. 2017) cuts the aligned score history into an even
number of contiguous slices, at most 16 (their illustrated `C(16, 8)`).
Every combination of half the slices is in-sample; the complement is
out-of-sample. Performance on a side is the mean score on those dates.
`pbo` is the fraction of combinations whose in-sample winner lands strictly
below the out-of-sample median. Ties are not counted as overfitting.

## Deflated and probabilistic scores, and the trial count

`psr` uses the Lo / Mertens standard error

```
SE = sqrt(1 - γ3 * SR + ((γ4 - 1) / 4) * SR²)
```

with `γ4` the raw fourth moment (a normal series has `γ4 = 3`). `dsr`
replaces the hurdle 0 with the expected maximum of `N` centered trials
(Euler–Mascheroni approximation in Bailey & López de Prado 2014).

`N` for `dsr` is `n_trials_effective`. Average-linkage clustering on the
Mantegna distance `sqrt((1-ρ)/2)` merges columns at correlation 0.5. That
threshold is fixed in advance (`CORRELATED_TRIAL_MIN_RHO`); it is not fit
to a result. Samples shorter than four observations are not clustered, and
every evaluated column counts. Configs with no aligned series each remain
their own trial, so dropping them cannot shrink the multiplicity.

`min_trl` inverts `psr` at 95% confidence against a hurdle of 0:

```
MinTRL = 1 + (1 - γ3 * SR + ((γ4 - 1) / 4) * SR²) * (z / SR)²
```

The 2012 paper's frequency table, for an annualized score of 2 against a
benchmark of 1 under normal moments, is 2.73 years of daily data (252),
2.83 weekly (52), and 3.24 monthly (12). With skewness -0.72 and raw
kurtosis 5.78 the monthly figure is 4.99 years.

The 2014 numerical example (annualized 2.5, 1,250 daily observations,
100 trials, variance `1/(2*250)`, skewness -3, raw kurtosis 10) has
`dsr ≈ 0.9004`. The same inputs at 46 trials, and normal moments at 88
trials, sit at about 0.9505.

## Receipt schema

Research notebooks are schema 2. Verification still accepts schema 1,
which has no overfitting block; those files are not rewritten.
`migrate_research_receipt` returns a schema-2 copy. When the block is
missing it stamps `metrics_status=legacy_uncomputed` and does not invent
probabilities. This schema is the research notebook's. The Phase-1
evidence index keeps its own schema.

A block that contains a forbidden headline key, a probability outside
[0, 1], an effective trial count above `n_trials`, or a `dsr` above `psr`
fails verification.
