.PHONY: help test test-full coverage lint typecheck doctor sync fmt security audit ci examples evidence docs docs-serve fx1-test fx1-lint fx1-corpus fx1-corpus-full fx1-eval fx1-gate

.DEFAULT_GOAL := help

help: ## Show targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

sync: ## Install the locked environment (all groups and extras)
	uv sync --frozen --all-groups --all-extras

test: ## PR-gate lab tests (not network, not slow; xdist)
	uv run pytest -n auto --dist loadfile -m "not network and not slow"

test-full: ## Full offline lab suite, including slow tests
	uv run pytest -n auto --dist loadfile -m "not network"

coverage: ## PR-gate tests + coverage (threshold in pyproject)
	# Threshold lives in [tool.coverage.report] (pyproject.toml) — no inline
	# --cov-fail-under so CI and local cannot drift. Sharded CI combines
	# partial data files and applies the same threshold once.
	uv run pytest -n auto --dist loadfile -m "not network and not slow" --cov --cov-report=term-missing --cov-report=xml

lint: ## Ruff check + format check on src/ and tests/
	uv run ruff check src tests
	uv run ruff format --check src tests

fmt: ## Auto-fix lint + format
	uv run ruff check --fix src tests
	uv run ruff format src tests

typecheck: ## mypy on the harness
	uv run mypy src/quant_fund

security: ## Bandit static security analysis on src/
	uvx --from bandit==1.9.4 bandit -q -r src --severity-level medium --confidence-level medium

audit: ## Locked-deps vulnerability audit (pip-audit)
	uv export --format requirements.txt --no-hashes --no-emit-project --all-extras --all-groups \
		| uvx --from pip-audit==2.10.1 pip-audit --strict -r /dev/stdin

doctor: ## Harness environment check
	uv run dipcatcher doctor

evidence: ## Regenerate docs/evidence/index.md from sealed receipts
	uv run python scripts/build_evidence_report.py

ci: lint typecheck coverage ## Local mirror of the CI gate

examples: ## Offline examples gallery: ruff, mypy, subprocess runner
	uv run ruff check examples tests/examples
	uv run ruff format --check examples tests/examples
	uv run mypy examples
	uv run pytest tests/examples -q

docs: ## Build the documentation site (strict)
	uv run --only-group docs --frozen mkdocs build --strict

docs-serve: ## Serve the documentation site locally
	uv run --only-group docs --frozen mkdocs serve --dev-addr 127.0.0.1:8000

# --- fx-1 (the model) lifecycle — dipcatcher is the harness ---------------
fx1-test: ## fx-1 test suite
	PYTHONPATH=src uv run pytest tests/fx1 -q

fx1-lint: ## fx-1 lint
	uv run ruff check src/fx1 tests/fx1

fx1-corpus: ## Build fx-1 SFT corpus from harness receipts + lab research runs
	# data/metadata/research/runs is host-local (gitignored, regenerable via
	# the lab research pipeline); the build degrades gracefully without it.
	uv run fx1 corpus build --receipts-dir receipts \
		--receipts-dir data/metadata/research/runs --out data/fx1/corpus.jsonl

fx1-corpus-full: ## Full corpus: receipts + research runs + notebooks + ledgers
	uv run fx1 corpus build-full --receipts-dir receipts \
		--receipts-dir data/metadata/research/runs --artifacts-dir artifacts \
		--notebooks docs/FX1.md --out data/fx1/corpus_full.jsonl

fx1-eval: ## Run eval task bank (requires MOONSHOT_API_KEY for hosted_k3)
	uv run fx1 eval --backend hosted_k3 --out data/fx1/eval.json

fx1-gate: fx1-lint ## Full fx-1 CI gate locally: lint + types + tests + honesty + corpus smoke
	uv run mypy src/fx1
	PYTHONPATH=src uv run pytest tests/fx1 -q
	PYTHONPATH=src uv run pytest tests/fx1 -q -k honesty
	uv run fx1 corpus build --receipts-dir receipts --out data/fx1/corpus_ci_smoke.jsonl
