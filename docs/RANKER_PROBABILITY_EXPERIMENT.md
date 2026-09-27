# Supervised ranker probability experiment

This research command asks whether a date-demeaned ridge ranker's score helps
forecast the sign of the next observed session's idiosyncratic return. It fits
Platt maps on a separate chronological calibration block. The controls are
Platt-calibrated `cs_pct_mom_20` and the training-block positive-label rate.
All three score the same date/name rows. No model is installed for forecasting.

Run it from a checkout with the locked environment:

```bash
uv sync --frozen --all-groups --all-extras
uv run python -m quant_fund.research.ranker_probability \
  --bronze-root data/file_us_wide/bronze \
  --lake-root /tmp/dipcatcher-ranker-probability-lake \
  --output /tmp/dipcatcher-ranker-probability.json
```

The command builds silver and gold through the existing offline data pipeline.
Alternatively pass existing `--features` and `--labels` gold Parquet paths.
The label file must carry `label_end_time_1`; the usual joined `panel()` drops
that endpoint. The JSON receipt records input and code hashes, ordered
features, split boundaries, label-maturity bounds, counts, date-weighted Brier
and log loss, paired stationary-block Brier intervals, and the predefined gate.
It is created exclusively; reruns need a fresh output path.

Each fold trains ridge on earlier dates, fits both Platt maps on later dates,
and scores an untouched subsequent test block. A whole decision date is
excluded from a fit when any included name's observed label endpoint reaches
the next block. A one-date embargo is also applied at both boundaries.
Training expands, calibration rolls forward, and test blocks are disjoint and
consecutive. The primary paired statistic first averages loss across names
within a test date, then takes ranker minus control. A seeded stationary
bootstrap of those ordered dates supplies the 95% percentile interval.

The statistical gate requires at least 250 scored test dates, no untrainable
folds, upper Brier confidence bounds below zero against both controls, and no
worse mean log loss against either. The receipt reports measurements when that
gate fails. `production_promotion` and `forward_evidence_accepted` remain false.

The checked-in Yahoo wide tape is a previously inspected, current-constituent
pool with reconstructed availability. Its costs and true historical membership
are not established. A result on this tape is exploratory and cannot establish
new unseen market evidence; SYNTHETIC tests establish code behavior only.

## Exploratory wide-tape result, 2026-09-27

The offline build produced 131,700 joined feature/label rows, of which 131,649
had a common finite feature and target set. There were 2,268 scored dates and
no untrainable folds. Equal-date Brier loss was **0.250215** for ranker Platt,
**0.250177** for momentum Platt, and **0.249953** for the training base rate.
The paired ranker-minus-momentum Brier interval was
[-0.000134, 0.000199]; ranker-minus-base-rate was [0.000122, 0.000407].
Ranker log loss was also higher than both controls. **The predefined gate
failed.** No model was promoted. The complete sealed diagnostic is
[`receipts/ranker_probability_wide_20260927.json`](receipts/ranker_probability_wide_20260927.json).
