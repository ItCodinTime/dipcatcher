# ADR-0008: Honesty is enforced by scanning artifact keys, not by convention

## Status

Accepted (discovered; documents existing behavior)

## Context

The honesty contract says research results are proper scores (pinball,
CRPS, PIT, QLIKE, Brier, ECE, Kupiec, HMM likelihood) — never headline
Sharpe/Sortino/Calmar/P&L/NAV. A convention alone is unenforceable: a
contributor adding `sharpe_mean` to a family blob, or a tool emitting
`nav` into a scorecard, would slip through review.

Instead the contract is **code that scans artifact structure**:

- `research/catalog/constants.py` `FORBIDDEN_RESEARCH_METRIC_KEYS` =
  `{sharpe, sortino, calmar, pnl, nav}`.
- `family_blob_forbidden_metrics_absent` walks every mapping key in a
  family blob, tokenizes on `_`/`-`, and fails closed when any token is
  forbidden. Scorecards in `run_research` record this as the
  `forbidden_metrics_absent` flag per family — the CI smoke requires it
  true for every required family.
- Scope is deliberate: paper `analytics_export` *may* carry `nav_*`/
  `*_pnl` diagnostics, so it's validated by `validate_analytics_export`
  (with `live_pnl_claim` fail-closed) instead — the same honesty enforced
  with the right tool per artifact type.
- The mirror on the model side, `fx1.honesty.FORBIDDEN_HEADLINE_TOKENS`,
  is text-level (regex on model *outputs*: "Sharpe 2.1" fails, discussing
  Sharpe passes) — and `tests/fx1/test_honesty_inheritance.py` fails CI
  if the two sets drift.
- Supplemental: hypothesis families are split `calibration`/`discovery`/
  `bound` and never pooled for BH-FDR (`format_fdr_families`); p-values
  must be in `[0,1]` or explicitly unavailable (null).

## Decision

Forbidden-metric detection operates on **keys** (structural tokens) in
research artifacts and on **headline patterns** (token + numeric value) in
fx-1 text — both fail closed, both mirrored, both tested.

## Consequences

- Honesty violations are caught by the artifact schema rather than code
  review — the CLI even prints `DATA_LABEL=` unconditionally
  (`format_data_label`).
- Values are never scanned, so a Sharpe *citation* or a `nav`-keyed input
  column is legal; the ban is on claiming, not mentioning.
- Renaming a metric to dodge the token list (`s_h_a_r_p_e` splits to
  different tokens) is only partially covered — a known residual risk
  mitigated by review plus the `leakage/` LH rules scanning source text.
- New headline metrics must be added to *both* forbidden sets; the
  inheritance test makes forgetting the fx1 side a CI failure.
