# ADR-0002: fx1 integrates through the receipt filesystem, not imports

## Status

Accepted (discovered; documents existing behavior)

## Context

The repo ships two packages — `quant_fund` (the dipcatcher harness) and
`fx1` (the model project). A naive design would have `fx1` call into
`quant_fund` for corpus building, honesty checks, or eval plumbing.

The generated module graph (`docs/architecture/module_deps.mmd`,
`docs/architecture/manifest.json`) shows **zero `fx1.* → quant_fund.*`
import edges**. The only cross-package edge is `quant_fund -> fx1`
(two statements in `quant_fund/__init__.py` re-exporting
`fx1.__version__`, which is also the hatch dynamic-version source —
single source of truth for the distribution version).

Instead, `fx1.data.corpus`/`receipts`/`ledgers` read lab *artifacts* —
`receipts/*.json`, `data/metadata/research/runs/` — hash them
(`receipt_sha256` round-trips into corpus rows), and treat ineligible
receipts as negative examples. `fx1` re-implements what it needs:
`fx1.honesty.FORBIDDEN_HEADLINE_TOKENS` mirrors
`quant_fund.research.catalog.FORBIDDEN_RESEARCH_METRIC_KEYS`, and
`tests/fx1/test_honesty_inheritance.py` blocks drift between them
(AGENTS.md requires they change together).

## Decision

The harness/model boundary is the **receipt filesystem contract plus a
drift-tested mirrored constant**, not a Python API.

## Consequences

- The two packages evolve independently; fx1 test lane does not import the
  lab suite's machinery and can gate on its own fixtures.
- Coupling is auditable data (receipt files) rather than call graphs — the
  same artifacts the honesty gates already seal.
- Risk accepted: shared semantics live in mirrored constants
  (`FORBIDDEN_*`) and duplicated small helpers; the inheritance test plus
  `docs/FX1_API_STABILITY.md` versioning carry the consistency burden.
- `quant_fund` depending on `fx1.__version__` (not the reverse) keeps the
  model project as the leaf: harness code cannot accidentally import model
  internals.
