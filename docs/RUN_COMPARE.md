# Run comparison (`quant_fund.research.compare`)

Paired comparison of two research runs — receipt JSONs (like the sealed blobs
under `receipts/`) or result directories containing `*.json` files.

## What it does

1. **Loads** each run. A directory with one JSON is loaded directly; a
   directory with several JSONs is merged under each file's stem
   (`part1.scores.x`, `part2.scores.x`, ...).
2. **Diffs configs**: config-like roots (`params`, `config`, `allocator`,
   `workload`, `environment`, `inputs_sha256`, `input_hashes`, `data`,
   `data_source`) plus every aligned scalar field (metric deltas).
3. **Aligns series**: every flat numeric list of length ≥ 2 is a
   per-observation score series (fold means, per-date losses, PIT histograms,
   latency samples, ...). Same-named series of equal length are paired on
   `delta = a - b`.
4. **Paired inference** per aligned series:
   - mean delta,
   - Diebold–Mariano test with Newey–West HAC correction (reuses
     `metrics.inference.diebold_mariano` — not reimplemented),
   - stationary-bootstrap percentile CI on the delta (Politis–Romano via
     `metrics.inference.stationary_bootstrap_indices`, block length from the
     Politis–White selector, seeded → deterministic and sign-symmetric).
5. **Honest verdicts** per series: `a_better` / `b_better` only when the DM
   p-value and the bootstrap CI agree; `no significant difference`;
   `no_effect` for identical series; `ambiguous` when DM and the CI disagree;
   `unaligned series lengths` for mismatched lengths; and
   **`insufficient paired observations`** when `n < min_paired` (default 10) —
   small-n runs are flagged, not silently tested.

## Honesty boundaries

- Loss convention by default: negative delta favors A. `--higher-is-better`
  flips the favored side; `delta = a - b` is always reported.
- Scalar fields whose key tokens match `FORBIDDEN_RESEARCH_METRIC_KEYS`
  (sharpe / sortino / calmar / pnl / nav) are moved to
  `excluded_diagnostics` — listed for context, never part of a verdict.
- Diagnostic only; no live-trading claims (`research_only: true`,
  `live_pnl_claim: false` on receipt output).

## CLI

```bash
# Markdown report to stdout
python -m quant_fund.research.compare receipts/run_a.json receipts/run_b.json

# JSON report to a file, plus a run_compare.v1 receipt blob
python -m quant_fund.research.compare a/ b/ \
    --format json --out compare.json --receipt-out compare_receipt.json

# Higher-is-better scores, stricter alpha, only one series
python -m quant_fund.research.compare a.json b.json \
    --higher-is-better --alpha 0.01 --series scores.pinball_per_fold
```

Options: `--alpha` (0.05), `--n-boot` (2000), `--seed` (7),
`--min-paired` (10), `--series` (restrict to named series),
`--format {markdown,json}`, `--out`, `--receipt-out`.

## Report contents

- **Verdicts** — per-verdict counts across aligned series.
- **Paired series** — n, means, delta, DM t/p, bootstrap CI, verdict.
- **Metric deltas** — aligned scalar fields that differ.
- **Excluded diagnostics** — forbidden-headline-token fields, context only.
- **Config diff** — leaf-level differences under config roots.
- **Unpaired series** — series present in only one run.

## API

```python
from quant_fund.research.compare import compare_runs

comp = compare_runs("run_a.json", "run_b.json")
print(comp.to_markdown())
payload = comp.to_dict()          # JSON-safe (NaN -> null)
```

## Limitations

- Pairing is positional within same-named equal-length series; it assumes both
  runs emitted observations in the same order (fold order, date order).
- With `n < ~30` the HAC t and bootstrap CI are both approximate; use
  `--min-paired` to force the honest insufficient-data verdict.
- Heterogeneous receipt schemas are flattened by path, so comparing runs from
  different schemas only aligns fields that share dotted paths.
