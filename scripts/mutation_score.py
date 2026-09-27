"""Mutation-test three numerical modules with mutmut and write the score.

Does not edit pyproject.toml. Writes a temporary setup.cfg, runs mutmut 3
against a narrow pytest selection, and stores the exported counts in
``tests/property/mutation_scores.json``. Invoke from the repo root:

    MLFLOW_DISABLE_AGENT_HINT=1 HYPOTHESIS_PROFILE=ci \\
        uv run --with mutmut python scripts/mutation_score.py

Score matches mutmut's badge formula: (killed + timeout) / (total - skipped).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "setup.cfg"
MUTANTS = ROOT / "mutants"
OUT = ROOT / "tests" / "property" / "mutation_scores.json"

# Narrow selections so each mutant reruns the tests that actually import it.
TARGETS: list[dict[str, object]] = [
    {
        "path": "src/quant_fund/execution/costs.py",
        "tests": [
            "tests/property/test_adversarial_costs.py",
            "tests/unit/test_execution.py",
        ],
    },
    {
        "path": "src/quant_fund/metrics/returns.py",
        "tests": [
            "tests/property/test_adversarial_returns_and_sizing.py",
            "tests/property/test_drawdown.py",
            "tests/unit/test_returns_edges.py",
            "tests/unit/test_metrics.py",
        ],
    },
    {
        "path": "src/quant_fund/utils/hashing.py",
        "tests": [
            "tests/property/test_adversarial_receipts.py",
            "tests/property/test_utils_invariants_cov.py",
            "tests/regression/test_receipt_set_and_array_canonical.py",
        ],
    },
]


def _setup_cfg(source: str, tests: list[str]) -> str:
    test_lines = "\n".join(f"    {name}" for name in tests)
    return (
        "[mutmut]\n"
        "source_paths =\n"
        f"    {source}\n"
        "also_copy =\n"
        "    src/quant_fund\n"
        "pytest_add_cli_args_test_selection =\n"
        f"{test_lines}\n"
        "pytest_add_cli_args =\n"
        "    -p\n"
        "    no:cacheprovider\n"
        "    --tb=line\n"
        "    -m\n"
        "    not network\n"
        "process_isolation = forkserver\n"
        "use_git_change_detection = false\n"
    )


def _score(stats: dict[str, int]) -> float | None:
    tested = int(stats.get("total", 0)) - int(stats.get("skipped", 0))
    if tested <= 0:
        return None
    killed = int(stats.get("killed", 0)) + int(stats.get("timeout", 0))
    return 100.0 * killed / tested


def _run_one(target: dict[str, object]) -> dict[str, object]:
    source = str(target["path"])
    tests = [str(name) for name in target["tests"]]  # type: ignore[union-attr]
    if MUTANTS.exists():
        shutil.rmtree(MUTANTS)
    SETUP.write_text(_setup_cfg(source, tests), encoding="utf-8")
    env = os.environ.copy()
    env.setdefault("HYPOTHESIS_PROFILE", "ci")
    env.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    start = time.perf_counter()
    run = subprocess.run(
        ["mutmut", "run", "--max-children", "4"],
        cwd=ROOT,
        env=env,
        check=False,
    )
    if run.returncode != 0:
        raise SystemExit(f"mutmut run failed for {source} (exit {run.returncode})")
    exported = subprocess.run(
        ["mutmut", "export-cicd-stats"],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if exported.returncode != 0:
        sys.stderr.write(exported.stdout)
        sys.stderr.write(exported.stderr)
        raise SystemExit(f"mutmut export-cicd-stats failed for {source}")
    stats_path = MUTANTS / "mutmut-cicd-stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    elapsed = time.perf_counter() - start
    version = subprocess.check_output(["mutmut", "--version"], cwd=ROOT, text=True).strip()
    return {
        "path": source,
        "tests": tests,
        "mutmut_version": version,
        "elapsed_sec": round(elapsed, 3),
        "killed": int(stats.get("killed", 0)),
        "survived": int(stats.get("survived", 0)),
        "timeout": int(stats.get("timeout", 0)),
        "suspicious": int(stats.get("suspicious", 0)),
        "skipped": int(stats.get("skipped", 0)),
        "no_tests": int(stats.get("no_tests", 0)),
        "total": int(stats.get("total", 0)),
        "score_percent": _score(stats),
        "score_formula": "(killed + timeout) / (total - skipped)",
    }


def main() -> None:
    if SETUP.exists():
        raise SystemExit(f"refusing to overwrite existing {SETUP}")
    rows: list[dict[str, object]] = []
    try:
        for target in TARGETS:
            print(f"=== {target['path']} ===", flush=True)
            row = _run_one(target)
            print(json.dumps(row, indent=2), flush=True)
            rows.append(row)
    finally:
        if SETUP.exists():
            SETUP.unlink()
        if MUTANTS.exists():
            shutil.rmtree(MUTANTS)
    payload = {
        "tool": "mutmut",
        "hypothesis_profile": os.environ.get("HYPOTHESIS_PROFILE", "ci"),
        "modules": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
