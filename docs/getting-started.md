# Getting started

Install the locked environment, confirm the lab is research-only, and verify a receipt before treating any score as evidence.

## Requirements

- Python 3.12 or newer (`requires-python` is `>=3.12`)
- [uv](https://docs.astral.sh/uv/)

`MOONSHOT_API_KEY` is used only by hosted fx-1 eval (`make fx1-eval`). The harness commands below run without it.

## Install

From the repository root:

```bash
make sync
```

That runs `uv sync --frozen --all-groups --all-extras`. `uv.lock` is authoritative. The `docs` dependency group (MkDocs) is included in `--all-groups` and is omitted from the runtime image, which syncs with `--no-dev`.

## Check the lab

```bash
uv run dipcatcher doctor --config configs/research.yaml
```

Proceed when `data_manifest` is `ok` and `live_allowed` is false for the research profile. The operations detail is in the [runbook](OPERATIONS_RUNBOOK.md).

## SYNTHETIC research smoke

```bash
uv run dipcatcher research --config configs/research.yaml
uv run dipcatcher verify-research
```

The default research config is labeled **SYNTHETIC**. The run checks that the engine, scorecard, and receipt verifier agree. It is a correctness test. It is not market evidence and it does not authorize promotion.

Real-file and public-source workflows are separate. Read [Data source labels](DATA_SOURCE_LABELS.md) before publishing a score, and [Receipt verification](RECEIPT_VERIFICATION.md) before citing one.

## This site

```bash
make docs
make docs-serve
```

`make docs` runs `mkdocs build --strict`. `make docs-serve` serves the site at `http://127.0.0.1:8000`.

## Next

- [Architecture](ARCHITECTURE.md)
- [Data contracts](DATA_CONTRACTS.md)
- [fx-1 overview](FX1.md)
- [API reference](api/index.md)
