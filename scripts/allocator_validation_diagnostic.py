"""Recheck allocator numerical reliability on the published validation tape.

This creates an unsealed retrospective diagnostic, never a tournament receipt.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.research import net_replay, real_benchmark
from quant_fund.research.net_tournament import _spec

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/metadata/cost_aware_tournament/us_wide_20260925"
OUTPUT = ROOT / "docs/research/allocator_numerical_diagnostic_20260927.json"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _published_validation() -> dict[str, Any]:
    with gzip.open(RUN / "validation.json.gz", "rt") as handle:
        value = json.load(handle)
    digest = value.get("receipt_sha256")
    if digest != real_benchmark._digest(
        {key: item for key, item in value.items() if key != "receipt_sha256"}
    ):
        raise ValueError("published validation receipt digest differs")
    return value


def _control(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": record["status"],
        "n_sessions": len(record["daily"]),
        "total_return": record["total_return"],
        "mean_daily_net_return": float(np.mean([row["net_return"] for row in record["daily"]])),
        "liquidation_complete": record["liquidation_complete"],
    }


def main() -> None:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise ValueError("commit source changes before stamping a diagnostic")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = real_benchmark._read_receipt(RUN / "manifest.json")
    benchmark_run = (RUN / manifest["benchmark_run"]).resolve()
    if (
        real_benchmark._read_receipt(benchmark_run / "manifest.json")
        != manifest["benchmark_manifest"]
    ):
        raise ValueError("published benchmark manifest differs")
    protocol = real_benchmark.protocol_for_run(manifest["benchmark_manifest"], benchmark_run)
    execution, trials, baseline = _spec(manifest["spec"])
    panel = net_replay.market_panel(
        real_benchmark._load_bars(protocol),
        open_column=manifest["spec"]["open_column"],
        volume_column=manifest["spec"]["volume_column"],
    )
    published = _published_validation()
    history = max(trial.required_history for trial in [baseline, *trials])
    report: dict[str, Any] = {
        "kind": "allocator_numerical_diagnostic",
        "sealed": False,
        "retrospective": True,
        "promote": False,
        "live_pnl_claim": False,
        "phase": "validation",
        "code_commit": revision,
        "allocator_source_sha256": _hash(ROOT / "src/quant_fund/research/cost_allocation.py"),
        "dataset_sha256": _hash(ROOT / "data/file_us_wide/bronze/bars.parquet"),
        "benchmark_config_sha256": _hash(ROOT / "configs/real_benchmark_us_wide.json"),
        "tournament_config_sha256": _hash(ROOT / "configs/cost_aware_tournament.json"),
        "sealed_validation_receipt_sha256": published["receipt_sha256"],
        "runtime": {
            "python": sys.version.split()[0],
            "cvxpy": importlib.metadata.version("cvxpy"),
            "clarabel": importlib.metadata.version("clarabel"),
            "numpy": np.__version__,
        },
        "solver_rules": {
            "accepted_status": "optimal",
            "max_original_constraint_violation": 1e-7,
            "max_solver_objective_gap": 1e-7,
            "clarabel_gap_and_feasibility_tolerances": 1e-8,
        },
        "scenarios": {},
        "limitations": [
            "Previously inspected, survivor-selected vendor snapshot with unverified adjustments and availability.",
            "This diagnostic is not a sealed tournament run and does not perform full-slate inference or validation selection.",
            "Return figures below are simulated price-return diagnostics, not live or independent forward performance.",
            "No held-out test data were read for this diagnostic.",
        ],
    }
    for scenario, multiplier in (("configured", 1.0), ("double_impact", 2.0)):
        controls = published["scenarios"][scenario]["trials"]
        record: dict[str, Any] = {
            "sealed_controls": {
                name: _control(controls[name])
                for name in ("equal_weight", "momentum_20", "reversal_1")
            },
            "repaired_allocators": {},
        }
        report["scenarios"][scenario] = record
        for trial in trials:
            if trial.allocation is None:
                continue
            try:
                replay = net_replay.replay(
                    panel,
                    trial,
                    execution,
                    start=protocol.validation_start,
                    end=protocol.validation_end,
                    history=history,
                    impact_multiplier=multiplier,
                )
                control_name = trial.name.removesuffix("_cost_aware")
                days = replay["daily"]
                if [row["date"] for row in days] != [
                    row["date"] for row in controls[control_name]["daily"]
                ] or [row["date"] for row in days] != [
                    row["date"] for row in controls["equal_weight"]["daily"]
                ]:
                    raise ValueError("repaired allocation calendar differs from controls")
                allocations = replay["allocations"]
                formulations = Counter(item["formulation"] for item in allocations)
                statuses = Counter(
                    attempt["status"] for item in allocations for attempt in item["solve_attempts"]
                )
                record["repaired_allocators"][trial.name] = {
                    "status": replay["status"],
                    "n_sessions": len(days),
                    "n_allocations": len(allocations),
                    "accepted_statuses": sorted({item["status"] for item in allocations}),
                    "accepted_formulations": dict(formulations),
                    "attempt_status_counts": dict(statuses),
                    "max_original_constraint_violation": max(
                        item["max_constraint_violation"] for item in allocations
                    ),
                    "max_solver_objective_gap": max(
                        item["solve_attempts"][-1]["solver_objective_gap"] for item in allocations
                    ),
                    "liquidation_complete": replay["liquidation_complete"],
                    "matched_rank_control": control_name,
                    "diagnostics": {
                        "total_simulated_net_price_return": replay["total_return"],
                        "mean_daily_net_difference_vs_rank_control": float(
                            np.mean(
                                [
                                    day["net_return"] - control["net_return"]
                                    for day, control in zip(
                                        days, controls[control_name]["daily"], strict=True
                                    )
                                ]
                            )
                        ),
                        "modeled_costs": {
                            key: float(sum(day[key] for day in days))
                            for key in ("commission", "spread", "impact", "borrow", "financing")
                        },
                    },
                }
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                record["repaired_allocators"][trial.name] = {
                    "status": "failed",
                    "error": str(exc),
                    "solver_diagnostic": getattr(exc, "diagnostic", None),
                }
            print(
                scenario,
                trial.name,
                record["repaired_allocators"][trial.name]["status"],
                flush=True,
            )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
