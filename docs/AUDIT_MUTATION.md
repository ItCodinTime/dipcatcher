# Mutation-testing audit — causality-critical modules

Mutation coverage (mutmut 3.8.0) for the fail-closed causality guards: purge /
embargo / CPCV / walk-forward splitting and the risk-overlay gate stack. Scores
and per-survivor dispositions live in `tests/property/mutation_scores.json`;
the kill tests live in `tests/unit/test_validation_branch_killers.py` and
`tests/unit/test_risk_branch_killers.py`. Reproduce with
`python scripts/mutation_score.py <module_stem>`.

| Module | Killed | Survived | Score | Survivors |
|---|---|---|---|---|
| `validation/purging.py` | 77 | 4 | 95.06% | all equivalent |
| `validation/embargo.py` | 33 | 3 | 91.67% | all equivalent |
| `validation/cpcv.py` | 289 | 49 | 85.50% | all equivalent |
| `validation/walk_forward.py` | 247 | 35 | 87.59% | all equivalent |
| `risk/overlay.py` | 372 | 27 | 93.23% | all equivalent |
| `risk/gates.py` | 826 | 89 | 90.27% | all equivalent |

Every surviving mutant was classified: each is either killed by an honest test
(asserting the real behavior difference, no tautologies) or listed in
`equivalent_survivors` with a one-line justification. The dominant equivalence
classes:

- **unreachable boundaries** — train indices can never equal a test-group
  boundary (`idx == lo`/`idx == hi` flips), `observe()` rejects non-positive
  navs so `peak <= 0`/`last <= 0` is dead, and guards like `n < n_groups` fire
  before the checked condition can hold;
- **value-preserving no-ops** — `dtype=None`/dropped `copy=`/`reshape(-2)` on
  float64/int64 arrays, `float("NAN")`, `zip(strict=True)` where an upstream
  check already enforces equal lengths, `uniq[None:]` == `uniq[0:]`;
- **unattainable exact equality** — `var == _EPS`, `base == _EPS`,
  `crc_es == _EPS`, `remaining == 0.005` were exhaustively searched and no
  float witness exists;
- **absorbed differences** — `min(…, 2.0)` clips erased by the final
  `min(scale, 1.0)`, `scaled/r ≡ scale` wherever `|r| > _EPS` holds, CRC's
  λ̂ ≥ ES so `allowed / es` never exceeds 1 on early clamped windows.

Notable kills this round: `x_overlaps` and `purge_mask` boundary flips,
`_one_timestamp_ns` float-precision loss for ns timestamps, walk-forward
holdout block-merge (`j == hi + 2` fabricates merged blocks and raises
spuriously), embargo/vol/ES/Kelly/CRC/crash/stepm gate-boundary flips, and
`apply_gate_stack` spec-forwarding drops (a silently-ignored `GateSpec` field
now fails a test that observes its effect).
