"""Mutation-test causality-critical modules with mutmut and write the score.

Does not edit pyproject.toml. Writes a temporary setup.cfg, runs mutmut 3
against a narrow pytest selection, and stores the exported counts in
``tests/property/mutation_scores.json``. Invoke from the repo root:

    MLFLOW_DISABLE_AGENT_HINT=1 HYPOTHESIS_PROFILE=ci \\
        uv run --with mutmut python scripts/mutation_score.py [stem ...]

With no arguments every target in TARGETS runs. Positional arguments filter
to targets whose module stem matches (e.g. ``purging gates``). Results merge
into the JSON by ``path``: rerunning one module replaces only its own entry,
and a hand-curated ``equivalent_survivors`` list is preserved across reruns.

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
            "tests/unit/test_cost_branch_killers.py",
        ],
    },
    {
        "path": "src/quant_fund/metrics/returns.py",
        "tests": [
            "tests/property/test_adversarial_returns_and_sizing.py",
            "tests/property/test_drawdown.py",
            "tests/unit/test_returns_edges.py",
            "tests/unit/test_metrics.py",
            "tests/unit/test_returns_branch_killers.py",
        ],
    },
    {
        "path": "src/quant_fund/utils/hashing.py",
        "tests": [
            "tests/property/test_adversarial_receipts.py",
            "tests/property/test_utils_invariants_cov.py",
            "tests/regression/test_receipt_set_and_array_canonical.py",
            "tests/unit/test_hashing_branch_killers.py",
        ],
    },
    {
        "path": "src/quant_fund/validation/purging.py",
        "tests": [
            "tests/unit/research/test_purging_edges.py",
            "tests/unit/research/test_cpcv_extremes.py",
            "tests/unit/pipeline/test_validation.py",
            "tests/unit/test_validation_branch_killers.py",
        ],
    },
    {
        "path": "src/quant_fund/validation/embargo.py",
        "tests": [
            "tests/unit/pipeline/test_embargo.py",
            "tests/unit/test_validation_branch_killers.py",
        ],
    },
    {
        "path": "src/quant_fund/validation/cpcv.py",
        "tests": [
            "tests/unit/research/test_cpcv_extremes.py",
            "tests/unit/research/test_backtest_overfitting.py",
            "tests/property/test_backtest_overfitting.py",
            "tests/unit/pipeline/test_validation.py",
            "tests/unit/test_validation_branch_killers.py",
        ],
    },
    {
        "path": "src/quant_fund/validation/walk_forward.py",
        "tests": [
            "tests/unit/pipeline/test_walk_forward_extremes.py",
            "tests/unit/research/test_cpcv_extremes.py",
            "tests/unit/research/test_fold_stability.py",
            "tests/unit/pipeline/test_validation.py",
            "tests/unit/test_validation_branch_killers.py",
        ],
    },
    {
        "path": "src/quant_fund/risk/overlay.py",
        "tests": [
            "tests/unit/risk/test_risk_gates.py",
            "tests/property/test_capacity_overlay.py",
            "tests/unit/research/test_capacity_overlay.py",
            "tests/unit/test_risk_branch_killers.py",
        ],
    },
    {
        "path": "src/quant_fund/risk/gates.py",
        "tests": [
            "tests/unit/risk/test_risk_gates.py",
            "tests/property/test_capacity_overlay.py",
            "tests/unit/research/test_capacity_overlay.py",
            "tests/unit/test_risk_branch_killers.py",
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
        "    src\n"
        "pytest_add_cli_args_test_selection =\n"
        f"{test_lines}\n"
        "pytest_add_cli_args =\n"
        "    -p\n"
        "    no:cacheprovider\n"
        "    --tb=line\n"
        "    -m\n"
        "    not network\n"
        "process_isolation = fork\n"
        "use_git_change_detection = false\n"
    )


def _score(stats: dict[str, int]) -> float | None:
    tested = int(stats.get("total", 0)) - int(stats.get("skipped", 0))
    if tested <= 0:
        return None
    killed = int(stats.get("killed", 0)) + int(stats.get("timeout", 0))
    return 100.0 * killed / tested


def _dump_survivors(source: str, env: dict[str, str]) -> None:
    """Optional: ``MUTMUT_SURVIVOR_DIR`` receives results and diffs before cleanup."""
    raw = os.environ.get("MUTMUT_SURVIVOR_DIR")
    if not raw:
        return
    dest = Path(raw)
    dest.mkdir(parents=True, exist_ok=True)
    listed = subprocess.run(
        ["mutmut", "results"],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    slug = Path(source).stem
    (dest / f"{slug}.results.txt").write_text(listed.stdout, encoding="utf-8")
    names: list[str] = []
    for line in listed.stdout.splitlines():
        if ":" not in line:
            continue
        name, status = line.rsplit(":", 1)
        if status.strip() == "survived":
            names.append(name.strip())
    chunks: list[str] = []
    for name in names:
        shown = subprocess.run(
            ["mutmut", "show", name],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        chunks.append(shown.stdout)
    (dest / f"{slug}.diffs.txt").write_text("\n".join(chunks), encoding="utf-8")


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
        ["mutmut", "run", "--max-children", os.environ.get("MUTMUT_MAX_CHILDREN", "4")],
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
    _dump_survivors(source, env)
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


def _select_targets(argv: list[str]) -> list[dict[str, object]]:
    if not argv:
        return list(TARGETS)
    wanted = {name.removesuffix(".py") for name in argv}
    chosen = [t for t in TARGETS if Path(str(t["path"])).stem in wanted]
    missing = wanted - {Path(str(t["path"])).stem for t in chosen}
    if missing:
        raise SystemExit(f"no TARGETS entries for: {sorted(missing)}")
    return chosen


def main() -> None:
    if SETUP.exists():
        raise SystemExit(f"refusing to overwrite existing {SETUP}")
    rows: list[dict[str, object]] = []
    try:
        for target in _select_targets(sys.argv[1:]):
            print(f"=== {target['path']} ===", flush=True)
            row = _run_one(target)
            print(json.dumps(row, indent=2), flush=True)
            rows.append(row)
    finally:
        if SETUP.exists():
            SETUP.unlink()
        if MUTANTS.exists():
            shutil.rmtree(MUTANTS)
    existing: dict[str, dict[str, object]] = {}
    prior: dict[str, object] = {}
    if OUT.exists():
        prior = json.loads(OUT.read_text(encoding="utf-8"))
        for entry in prior.get("modules", []):
            existing[str(entry["path"])] = entry
    for row in rows:
        previous = existing.get(str(row["path"]), {})
        curated = previous.get("equivalent_survivors") or []
        row["equivalent_survivors"] = curated
        existing[str(row["path"])] = row
    payload: dict[str, object] = {k: v for k, v in prior.items() if k != "modules"}
    payload["tool"] = "mutmut"
    payload["hypothesis_profile"] = os.environ.get("HYPOTHESIS_PROFILE", "ci")
    payload["modules"] = list(existing.values())
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
