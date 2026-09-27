.PHONY: help test coverage lint typecheck doctor sync fmt security audit ci examples evidence native docs docs-serve formal simtest simtest-large fx1-test fx1-lint fx1-corpus fx1-corpus-full fx1-eval fx1-gate mc-engine-smoke diffbacktest proofcore-test proofcore-coverage proof-integrity proof-verify leakage-scan reality-gate receipts-reverify pretrade-bench

.DEFAULT_GOAL := help

help: ## Show targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

sync: ## Install the locked environment (all groups and extras)
	uv sync --frozen --all-groups --all-extras

test: ## Lab test suite (unit/property/regression/end_to_end)
	uv run pytest

coverage: ## Lab tests + coverage (threshold in pyproject [tool.coverage.report])
	# Threshold lives in [tool.coverage.report] (pyproject.toml) — no inline
	# --cov-fail-under so CI and local cannot drift.
	uv run pytest -m "not network" --cov --cov-report=term-missing --cov-report=xml

lint: ## Ruff check + format check on src/ and tests/
	uv run ruff check src tests
	uv run ruff format --check src tests

fmt: ## Auto-fix lint + format
	uv run ruff check --fix src tests
	uv run ruff format src tests

typecheck: ## mypy on the harness
	uv run mypy src/quant_fund

diffbacktest: ## Differentiable backtest (optional JAX extra, CPU)
	JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" uv run pytest -q tests/unit/diffbacktest

security: ## Bandit static security analysis on src/
	uvx --from bandit==1.9.4 bandit -q -r src --severity-level medium --confidence-level medium

audit: ## Locked-deps vulnerability audit (pip-audit)
	uv export --format requirements.txt --no-hashes --no-emit-project --all-extras --all-groups \
		| uvx --from pip-audit==2.10.1 pip-audit --strict -r /dev/stdin

doctor: ## Harness environment check
	uv run dipcatcher doctor

native: ## Build optional quant_core (Rust + maturin). NumPy stays the fallback.
	uv pip install "maturin>=1.7,<2"
	uv run maturin develop --release --manifest-path rust/quant_core/Cargo.toml

evidence: ## Regenerate docs/evidence/index.md from sealed receipts
	uv run python scripts/build_evidence_report.py

ci: lint typecheck coverage ## Local mirror of the CI gate

formal: ## TLC order-lifecycle check + Z3/conformance/stateful tests
	bash scripts/run_tlc.sh
	uv run pytest tests/formal -q

mc-engine-smoke: ## Monte Carlo engine tests (not slow) and a tiny CLI run
	uv run pytest tests/unit/mc_engine -q -m "not slow"
	uv run python -m quant_fund.mc_engine run --paths 1500 --steps 8 --workers 1 \
		--backend serial --chunk-size 500 --seed 1 --no-progress

pretrade-bench: ## Pre-trade hot-path latency gate (p50 < 5us, p99 < 20us)
	uv run python -m quant_fund.pretrade.bench --gate

examples: ## Offline examples gallery: ruff, mypy, subprocess runner
	uv run ruff check examples tests/examples
	uv run ruff format --check examples tests/examples
	uv run mypy examples
	uv run pytest tests/examples -q

docs: ## Build the documentation site (strict)
	uv run --only-group docs --frozen mkdocs build --strict

docs-serve: ## Serve the documentation site locally
	uv run --only-group docs --frozen mkdocs serve --dev-addr 127.0.0.1:8000

simtest: ## Bounded deterministic-simulation tests and swarm (CI size)
	uv run pytest tests/unit/simtest tests/regression/test_simtest_duplicate_bar_order.py -q -m "not slow"
	MLFLOW_DISABLE_AGENT_HINT=1 uv run python scripts/simtest_swarm.py --seeds 64 --days 5 --base-seed 0

simtest-large: ## Large seeded swarm (workflow_dispatch size; not the PR default)
	MLFLOW_DISABLE_AGENT_HINT=1 uv run python scripts/simtest_swarm.py --seeds 4000 --days 8 --base-seed 0

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

# --- PROOFCORE gates (DESIGN.md §9.3; additive — no existing target changed) -
# Merge order: W5 lands last; the W1-W4 CLIs below exist once those waves merge.

PROOFCORE_LEDGER ?= data/metadata/proofcore-trials.jsonl
PROOFCORE_DB ?= data/metadata/proofcore.duckdb

proofcore-test: ## PROOFCORE W5 tests: contracts, provenance DB, CI helpers, layering gate
	uv run pytest tests/unit/test_proofcore_*.py tests/end_to_end/test_proofcore_smoke.py -q

proofcore-coverage: ## Per-package coverage floors (A3 #2): pit/proof/reality/proofcore 90, leakage 85
	# Subset run over the PROOFCORE test lanes; the global 80% floor still
	# applies to the full `make coverage` lane and is NOT re-asserted here
	# --cov-fail-under=0 defers to the per-package gates below, which read
	# [tool.proofcore.coverage-floors] from pyproject.toml).
	uv run pytest -m "not network" -q \
		-k "pit or proof or leakage or reality or proofcore" \
		--cov --cov-report= --cov-fail-under=0
	uv run python -m quant_fund.proofcore.ci coverage-gate

proof-integrity: ## Current proof signer/recorder integrity checks
	uv run pytest tests/unit/proof/test_integrity.py -q

proof-verify: ## Bundle hash, sidecar, signature, and metric verification tests; replay remains closed
	uv run pytest tests/unit/test_proof_bundle.py tests/unit/test_proof_verify.py tests/property/test_backtest_receipt_identity.py -q

leakage-scan: ## Leakage hunter — WARN MODE this wave (adjudicated: advisory only)
	@echo "leakage-scan is WARN MODE this wave: findings are advisory, the gate"
	@echo "flips to --fail-on error in the migration wave (DESIGN.md §13 phase 3)."
	uv run quant leakage scan --paths src/quant_fund --format json > leakage-report.json || \
		echo "::warning::leakage scan failed or is not yet merged (advisory this wave)"
	@if [ -d tests/leakage_fixtures ]; then \
		uv run pytest tests/unit -q -k "leakage"; \
	else \
		echo "tests/leakage_fixtures not present yet (W3 lands separately); skipped"; \
	fi

reality-gate: ## Reality-filter gate: export trial ledger from provenance DB + ledger-gate
	uv run quant proofcore export --db $(PROOFCORE_DB) --out $(PROOFCORE_LEDGER)
	uv run quant reality trial-report --ledger $(PROOFCORE_LEDGER)
	uv run quant reality ledger-gate --ledger $(PROOFCORE_LEDGER)

receipts-reverify: ## Fail-closed audit; schema-specific committed receipt verifiers pending
	uv run python -m quant_fund.proofcore.ci receipts-reverify receipts
