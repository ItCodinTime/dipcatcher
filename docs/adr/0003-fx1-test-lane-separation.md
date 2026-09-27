# ADR-0003: `tests/fx1` lives outside the default pytest testpaths

## Status

Accepted (discovered; documents existing behavior)

## Context

Two test suites exist: the lab suite (`tests/unit`, `tests/property`,
`tests/regression`, `tests/end_to_end`, ~700 files) and the fx-1 suite
(`tests/fx1`, ~200 tests per AGENTS.md). `pyproject.toml` sets

```toml
testpaths = ["tests/unit", "tests/property", "tests/regression", "tests/end_to_end"]
```

so a bare `pytest` (and `make test`, and the CI `test` job) never collects
`tests/fx1`. fx-1 runs in its own lane: `make fx1-test` sets
`PYTHONPATH=src`, `make fx1-gate` runs lint+mypy+tests+honesty+corpus smoke,
and `.github/workflows/fx1.yml` is a separate workflow — even running the
honesty-inheritance subset as its own blocking step. AGENTS.md documents
the intent explicitly, including that a dropped `tests/tests` mirror must
stay out of default collection.

## Decision

Keep the fx-1 suite a **separate, opt-in test lane** rather than a testpath.

## Consequences

- The lab suite stays hermetic: `make test` cannot be slowed or broken by
  model-lane dependencies (torch `nn` extra, corpus fixtures).
- fx-1 gets its own gate contract — the honesty-inheritance run
  (`pytest tests/fx1 -k honesty`) is a *blocking* step in `fx1.yml`,
  matching the documented contract in `docs/FX1_TRAINING.md`.
- Cost: a contributor running bare `pytest` sees green while fx-1 is
  broken; mitigated by `make fx1-gate`/`fx1.yml` being the documented
  model-lane gate and by AGENTS.md calling out the exclusion "on purpose".
- Contributors must not re-add a `tests/tests` mirror or move fx1 tests
  under default collection without a conscious ADR-level decision.
