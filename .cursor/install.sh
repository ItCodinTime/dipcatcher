#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for Dipcatcher.
# Installs uv (if missing) and syncs the locked virtualenv with dev groups and
# all extras (torch) so lint, type-check, the full test suite, the CLI, and the
# FastAPI service are all runnable out of the box.
set -euo pipefail

# 1. Ensure uv is available (pinned major line; installer is a no-op if present).
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# Make uv visible in this shell regardless of prior PATH state.
# shellcheck disable=SC1090
[ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
export PATH="$HOME/.local/bin:$PATH"

uv --version

# 2. Sync the locked environment. --all-groups pulls dev tooling (pytest, ruff,
#    mypy); --all-extras pulls the optional `nn` extra (torch) required by the
#    deep-RL benches and their tests. --frozen fails if uv.lock is stale.
uv sync --frozen --all-groups --all-extras
