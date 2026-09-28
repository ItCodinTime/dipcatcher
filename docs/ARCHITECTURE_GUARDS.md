# Architecture guards

`scripts/check_import_boundaries.py` is a stdlib-only (`ast` + `tomllib`)
enforcement check that fails CI when `src/` code violates the declared import
boundaries. The rule set lives in `configs/arch_boundaries.toml`; the gate is
`.github/workflows/arch_guards.yml` (job `import-boundaries`).

```bash
# run the gate locally (no env needed beyond Python 3.12)
uv run --no-project python scripts/check_import_boundaries.py
# see which violations the baseline currently suppresses
uv run --no-project python scripts/check_import_boundaries.py --show-baselined
# checker self-tests (also run in the workflow)
uv run --no-project python -m unittest tests.unit.arch.test_import_boundaries
```

Exit codes: `0` clean, `1` violations (or stale baseline entries), `2`
config/usage error.

## The layer rules

`quant_fund`'s 47 top-level packages are assigned to eight ordered layers. A
module may import its own layer and anything below it; importing a higher
layer at module scope is a `layer-order` violation.

| Layer | Packages | Role |
|---|---|---|
| `foundation` | schemas, config, utils, registry, compute, proofcore, hmm, observe, calendars | contracts and primitives; no intra-repo deps |
| `market_data` | data, pit, labels, features, fusion, microstructure, northset, metrics | ingest, PIT vault, features, LOB estimators, shared scoring vocabulary |
| `analytics` | models, portfolio, quant_models, risk, monitoring, diffbacktest, mc_engine, lightspeed, native, reporting, validation | models + allocation + the tools that score them |
| `execution` | execution, pretrade, parity, formal | simulated broker, cost models, pre-trade and parity checks |
| `orchestration` | pipeline, backtest | the train/forecast and replay engines that research drives |
| `research` | research, robustness, leakage, audit, stress, proof, reality | research catalog/benches and its verification tooling |
| `simulation` | hedge_lab, market_sim, paper | books/sessions that consume research outputs and the engines |
| `interface` | api, cli, simtest | entry points |

Two deliberate exemptions:

* **The facade.** `quant_fund` (`__init__.py`) and `quant_fund.public` are the
  stable public surface — they wire across layers by design and are
  importable by anything. `[[deny]]` rules still apply to them.
* **Lazy imports.** An import inside a function body is exempt from
  `layer-order` — deferred imports are this codebase's sanctioned mechanism
  for breaking cycles (e.g. `cli` lazily loading `paper`, `utils` lazily
  loading `native`). `[[deny]]` and `[[allow_only]]` rules **do** apply to
  lazy and `TYPE_CHECKING` imports, so the safety boundaries cannot be
  bypassed by moving an import into a function.

A package that no layer claims is itself a violation
(`unclassified-package`): when you add a top-level package under
`quant_fund`, you must add it to a layer in the config — that is the point.

## Hard rules (`[[deny]]`, `[[allow_only]]`)

These encode the repo's hard rules mechanically, on **every** import site:

* `library-no-cli-imports` — nothing outside `quant_fund.cli` may import it
  (mirrors `tests/unit/test_architecture.py`).
* `order-path-boundary` — `quant_fund.paper.*` and
  `execution.simulated_broker` are reachable only from `cli`, `paper`,
  `execution`, `simtest`, `pretrade`, `formal`, and `parity`. Research may
  still use `execution.*` cost/impact models (e.g. `almgren_chriss`). This is
  the mechanical version of the repo rule "research code must not import
  live/broker modules", and the importer whitelist mirrors
  `BROKER_ALLOWED_PREFIXES` in `tests/unit/test_architecture.py` — keep the
  two in sync.
* `research-purity` — `quant_fund.research.*` may not import `api`,
  `monitoring`, `observe`, or `pretrade`, even lazily.
* `harness-no-fx1-imports` — `quant_fund` must not import `fx1`; the single
  exception is `quant_fund/__init__.py` reading `fx1.__version__` (hatch
  derives the distribution version from it — an intentional seam).
* `proofcore-standalone` — `proofcore` may not import the rest of
  `quant_fund` (documented invariant in `proofcore/contracts.py`).
* `fx1-harness-surface` (`allow_only`) — `fx1` may import only
  `quant_fund.{config, data, metrics, schemas, utils, validation}`. Any new
  coupling to pipeline/research/cli/api/paper fails.

## Reading violations

```
src/quant_fund/data/x.py:12: error[layer-order]: quant_fund.data.x (layer
  'market_data') imports quant_fund.research.y (layer 'research') — lower
  layers must not depend on higher layers
src/quant_fund/research/z.py:40: error[order-path-boundary]: ... [lazy import]
configs/arch_boundaries.toml: error[stale-baseline]: entry file=... matches
  no violation — remove the entry
```

Each line names the file:line, the imported module, and the violated rule.
`[lazy import]` / `[TYPE_CHECKING]` flags mark the import context.

## The baseline

`[[baseline]]` entries itemize pre-existing violations so CI is green while
the debt is paid down. **Current count: 10** (all `layer-order`):

| File | Imports | Why |
|---|---|---|
| `metrics/cross_section.py` | `models.asset_pricing` | scoring reaches into a model family |
| `metrics/analytics.py` | `portfolio.attribution` | scoring reaches into portfolio |
| `models/robinhood_plus/compare.py` | `pipeline.dataset`, `pipeline.forecast`, `research.catalog` | a benchmark harness lives under `models/` and drives the pipeline |
| `models/robinhood_plus/bench.py` | `research.catalog` | same harness |
| `northset/benches.py` | `research.catalog` | bench adapters read the catalog |
| `pipeline/doctor.py` | `research.catalog`, `research.verify` | doctor inspects research receipts |
| `validation/gates.py` | `research.verify` | the gate sits above its package's layer |

How to update it:

* **Fix the import, then delete the entry.** A baseline entry that matches
  nothing becomes a `stale-baseline` error — the list cannot rot.
* **Never baseline new code.** Entries are only for debt that pre-dates this
  gate; new violations must be fixed in the PR that introduces them.
* Entry fields: `file` (repo-relative path), `module` (exact module or a
  `.**` prefix pattern), `rule` (`layer-order`, a deny name, or `*`), and a
  `note` naming an owner.

## Scope and limitations

* Static `ast` analysis only — it sees import statements, not
  `importlib`/`__import__` dynamic loads.
* Layers are per-package granularity (first module segment under
  `quant_fund`); finer-grained boundaries inside a package need
  `[[deny]]`/`[[allow_only]]` rules.
* `layer-order` deliberately ignores function-level imports; a lazy import
  can still violate `[[deny]]` rules.
* `tests/unit/test_architecture.py` keeps its own hardcoded checks (module
  size, no-cli-imports, broker/paper boundary); this config is a superset —
  same invariants, declared in one place, plus the full layer order.
