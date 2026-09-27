# Model explainability reports

`quant_fund.research.explainability` produces feature-attribution reports for
any fitted dipcatcher forecast head — anything that satisfies the
`ForecastModel` contract in `quant_fund.models.base` (`fit(x, y)` /
`predict(x)` / `metadata()`), or a bare `predict` callable.

Research-only tooling. Attribution is always measured as the degradation of a
**proper score** — pinball, CRPS, or Brier — never accuracy, Sharpe, P&L, or
NAV (repo honesty contract). Reports are diagnostic artifacts, not promotion
gates and not evidence of live profitability.

## What it computes

1. **Permutation importance** (`permutation_attribution`): each feature column
   is shuffled `n_repeats` times under a seeded `numpy.random.Generator`, and
   the mean *increase* in the proper loss is reported per feature. Same
   estimator protocol as `sklearn.inspection.permutation_importance`
   (Breiman 2001), implemented directly so a bare `predict` callable works and
   the permutation stream is transparently deterministic under `seed`.
2. **Partial dependence** (`partial_dependence_1d` / `_top_k`): 1-D curves on
   an empirical-quantile grid, computed directly — no extra dependency. For
   quantile heads the mean over output columns is reported per grid point.
3. **Attribution drift** (`attribution_drift`): the evaluation window is split
   into contiguous time-ordered blocks (sorted by `times`; row order if
   absent), importance is recomputed per block with the model held fixed, and
   drift is summarized by consecutive-block Spearman rank correlation,
   Jensen–Shannon divergence on ε-smoothed importance shares (nats,
   `≤ ln 2`), and top-k overlap. Drift is flagged when
   `min_spearman < spearman_floor` (0.6), `max_js > js_ceiling` (0.1), or
   `mean_topk_overlap < topk_floor` (0.5) — tunable heuristics, not gates.
4. **Optional SHAP path** (`shap_attribution`): `shap` (added as a project
   dependency, `shap>=0.52`) is imported lazily; the permutation explainer
   yields mean-|SHAP| importances as a supplementary view. It is labeled as
   prediction-space, not a proper-loss decomposition. If `shap` is missing,
   the permutation path is unaffected and `build_report` records a warning.

## Proper scores only

```python
from quant_fund.research.explainability import pinball, crps_quantiles, brier

pinball(0.5)          # default; a point forecast is the τ-quantile forecast
crps_quantiles((0.05, 0.25, 0.5, 0.75, 0.95))  # heads emitting (n, k) quantile grids
brier()               # probability heads emitting P(y=1)
```

`build_report(..., scoring=...)` also accepts `"pinball:0.5"`,
`"crps:0.1,0.5,0.9"`, `"brier"`, or a custom `(y, pred) -> float` callable —
custom scores are the caller's honesty responsibility and must be
lower-is-better proper losses.

## Usage

```python
from quant_fund.models.ranking import RidgeRanker
from quant_fund.research.explainability import build_report, write_report

model = RidgeRanker(alpha=1.0).fit(x_train, y_train)
report = build_report(
    model, x_eval, y_eval,
    times=eval_dates,        # optional; enables time-ordered drift blocks
    feature_names=features,  # optional; falls back to model.metadata().features
    scoring="pinball:0.5",
    n_blocks=4, n_repeats=5, seed=42,
    synthetic=True,          # required label for synthetic eval data
)
paths = write_report(report, "out/explainability")
# -> explainability.md, explainability.html, explainability.json (+ .sha256)
```

- **Markdown** — importance table, PD tables, drift statistics.
- **HTML** — self-contained (charts are base64 PNGs drawn with matplotlib's
  Agg canvas; no external assets or JavaScript).
- **JSON** — machine-readable `explainability_report.v1` payload, fit for
  catalog/bench tooling.

Determinism: `build_report` with the same `seed`, data, and model is fully
reproducible (the JSON payload is byte-identical modulo `generated_at`).

## Receipt extension (optional, additive, backward-compatible)

Sealed research receipts are immutable — `verify_research_artifact` requires
`latest.json` to byte-match `runs/<run_id>.json` — so the extension never
touches the receipt file. Instead a sidecar is written next to it:

```
data/metadata/research/latest.json                  (sealed receipt — untouched)
data/metadata/research/explainability/…             (report files)
data/metadata/research/latest.explainability.json   (sidecar binding)
```

```python
from quant_fund.research.explainability import (
    attach_explainability_report, verify_explainability_sidecar,
)

paths = attach_explainability_report(receipt_path, report)
binding = verify_explainability_sidecar(receipt_path)
assert binding["valid"]
```

The sidecar (`explainability_attachment.v1`) binds the parent receipt by
`sha256` + `run_id` and lists each report artifact's `sha256`/size/kind.
`verify_explainability_sidecar` re-checks the binding; it is a separate,
additive check — `verify_research_artifact` neither knows about nor needs the
sidecar, and sealed receipts verify identically with or without it.

For receipts minted in the future, `explainability_artifact_entry(report_dir,
receipt_dir)` returns an optional `artifacts.explainability` mapping
(relative paths + sha256s) that a receipt writer may embed **at write time** —
the verifier ignores unknown artifact keys, so it stays backward-compatible.
Retrofitting it onto an already-sealed receipt remains forbidden (it would
break `immutable_receipt_mismatch`).

## Limitations

- Permutation importance is associational, not causal; correlated features
  split credit arbitrarily (standard caveat — report discloses it).
- Drift thresholds are diagnostics heuristics for research review, not
  promotion gates; small blocks make Spearman noisy on near-tied tails.
- The SHAP path is prediction-space mean-|value|; it is a supplementary
  ranking view, not a proper-score decomposition.
- PD curves are 1-D marginals and can mislead under strong feature
  interactions; read them next to the drift heatmap, not alone.
