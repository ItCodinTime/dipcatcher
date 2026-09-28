# ADR-0002: cross-root imports are pinned; corpus stays on receipts

## Status

Accepted (discovered; documents existing behavior)

## Context

The repo ships two packages — `quant_fund` (the dipcatcher harness) and
`fx1` (the model project). Corpus construction, honesty checks, and the
receipt ledger do not call into each other.

`fx1.data.corpus` / `receipts` / `ledgers` read lab *artifacts* —
`receipts/*.json`, `data/metadata/research/runs/` — hash them
(`receipt_sha256` round-trips into corpus rows), and treat ineligible
receipts as negative examples. `fx1.honesty.FORBIDDEN_HEADLINE_TOKENS`
mirrors `quant_fund.research.catalog.FORBIDDEN_RESEARCH_METRIC_KEYS`, and
`tests/fx1/test_honesty_inheritance.py` blocks drift between them
(AGENTS.md requires they change together).

The generated module graph (`docs/architecture/module_deps.mmd`,
`docs/architecture/manifest.json`) is checked by
`tests/unit/docs/test_arch_atlas.py`:

- `quant_fund` → `fx1` is one edge: `quant_fund/__init__.py` imports
  `fx1.__version__` (the hatch dynamic-version source). A destination of
  `fx1.data.corpus` or any other `fx1.*` module is still a cross-root edge
  and fails the pin. Matching only the exact name `fx1` would miss it.
- `fx1.forecast` and `fx1.eval` import a fixed set of harness modules
  (scoring, point-in-time validation, hashing, schemas, walk-forward).
  Any other `fx1.*` → `quant_fund.*` edge fails the same pin.

## Decision

Corpus and honesty stay on the **receipt filesystem plus a drift-tested
mirrored constant**. The forecast harness may call only the pinned
helpers. The lab may import `fx1` only to re-export `__version__`.

## Consequences

- Adding a cross-root import fails `test_real_graph_shape_invariants`
  until the pin and this record are updated together.
- Coupling of the corpus is auditable data (receipt files). The forecast
  harness's coupling is the pinned edge list, not an open import policy.
- Risk accepted: shared honesty semantics live in mirrored constants
  (`FORBIDDEN_*`); the inheritance test plus `docs/FX1_API_STABILITY.md`
  carry that consistency burden.
- Harness code cannot import model internals. The version re-export is the
  only `quant_fund` → `fx1` edge.
