.PHONY: test coverage lint typecheck doctor sync fmt security audit ci fx1-test fx1-lint fx1-corpus fx1-corpus-full fx1-eval

sync:
	uv sync --frozen --all-groups

test:
	uv run pytest

coverage:
	# Threshold lives in [tool.coverage.report] (pyproject.toml) — no inline
	# --cov-fail-under so CI and local cannot drift.
	uv run pytest -m "not network" --cov --cov-report=term-missing --cov-report=xml

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

fmt:
	uv run ruff check --fix src tests
	uv run ruff format src tests

typecheck:
	uv run mypy src/quant_fund

security:
	uvx --from bandit==1.9.4 bandit -q -r src --severity-level medium --confidence-level medium

audit:
	uv export --format requirements.txt --no-hashes --no-emit-project --all-extras --all-groups \
		| uvx --from pip-audit==2.10.1 pip-audit --strict -r /dev/stdin

doctor:
	uv run dipcatcher doctor

ci: lint typecheck coverage

# --- fx-1 (the model) lifecycle — dipcatcher is the harness ---------------
fx1-test: ## fx-1 test suite
	PYTHONPATH=src uv run pytest tests/fx1 -q

fx1-lint: ## fx-1 lint
	uv run ruff check src/fx1 tests/fx1

fx1-corpus: ## Build fx-1 SFT corpus from harness receipts + lab research runs
	uv run fx1 corpus build --receipts-dir receipts \
		--receipts-dir data/metadata/research/runs --out data/fx1/corpus.jsonl

fx1-corpus-full: ## Full corpus: receipts + research runs + notebooks + ledgers
	uv run fx1 corpus build-full --receipts-dir receipts \
		--receipts-dir data/metadata/research/runs --artifacts-dir artifacts \
		--notebooks docs/FX1.md --out data/fx1/corpus_full.jsonl

fx1-eval: ## Run eval task bank (requires MOONSHOT_API_KEY for hosted_k3)
	uv run fx1 eval --backend hosted_k3 --out data/fx1/eval.json
