# ADR-001: uv and Python 3.12

## Status

Accepted

## Date

2026-09-15

## Context

The platform needs a pinned, reproducible Python environment. System Python on the workstation is 3.9. hmmlearn documents wheels for 3.8–3.12.

## Options considered

- Poetry / pip-tools on 3.11+
- uv with Python 3.12
- uv with Python 3.13

## Decision

Use **uv** with **Python 3.12** (`requires-python = ">=3.12"`, `.python-version` 3.12). Commit `uv.lock`.

## Consequences

3.13 is not the CI target until hmmlearn (or a replacement) supports it. Contributors install uv and run `uv sync`.
