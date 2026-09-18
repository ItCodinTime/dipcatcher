.PHONY: test coverage lint typecheck doctor sync fmt security audit ci

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
