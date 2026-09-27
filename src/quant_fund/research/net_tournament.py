"""Freeze and compare net returns on real_benchmark's declared data/splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.metrics import inference, snooping
from quant_fund.research import cost_allocation, net_replay, real_benchmark
from quant_fund.research.cost_allocation import AllocationConfig, AllocationFailure
from quant_fund.research.net_replay import ReplayConfig, Strategy, market_panel, replay
from quant_fund.research.real_benchmark import (
    _load_bars,
    _read_receipt,
    _runtime,
    _seal,
)


def _code_hashes() -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path(__file__),
            Path(net_replay.__file__),
            Path(cost_allocation.__file__),
            Path(real_benchmark.__file__),
            Path(inference.__file__),
            Path(snooping.__file__),
        )
    }


def _tournament_runtime() -> dict[str, Any]:
    import importlib.metadata

    return {
        **_runtime(),
        **{name: importlib.metadata.version(name) for name in ("cvxpy", "clarabel", "scipy")},
    }


def allocation_ablations(outcomes: dict[str, Any], trials: list[Strategy]) -> list[dict[str, Any]]:
    """Descriptive matched differences, with no unadjusted significance claim."""
    reports = []
    costs = ("commission", "spread", "impact", "borrow", "financing")
    for candidate in trials:
        if candidate.allocation is None:
            continue
        for control in trials:
            if replace(candidate, name=control.name, allocation=None) != control:
                continue
            a, b = outcomes[candidate.name], outcomes[control.name]
            row: dict[str, Any] = {
                "candidate": candidate.name,
                "control": control.name,
                "inference": "descriptive_only",
            }
            if a["status"] != "completed" or b["status"] != "completed":
                reports.append({**row, "status": "incomplete_pair"})
                continue
            if [d["date"] for d in a["daily"]] != [d["date"] for d in b["daily"]]:
                raise ValueError("matched allocation calendars differ")
            reports.append(
                {
                    **row,
                    "status": "completed",
                    "mean_daily_net_difference": float(
                        np.mean(
                            [
                                x["net_return"] - y["net_return"]
                                for x, y in zip(a["daily"], b["daily"], strict=True)
                            ]
                        )
                    ),
                    "total_return_difference": a["total_return"] - b["total_return"],
                    "max_drawdown_difference": a["max_drawdown"] - b["max_drawdown"],
                    "total_cost_difference": sum(sum(d[k] for k in costs) for d in a["daily"])
                    - sum(sum(d[k] for k in costs) for d in b["daily"]),
                    "traded_notional_difference": sum(
                        abs(f["quantity"]) * f["price"] for f in a["fills"]
                    )
                    - sum(abs(f["quantity"]) * f["price"] for f in b["fills"]),
                    "both_liquidated": a["liquidation_complete"] and b["liquidation_complete"],
                }
            )
    return reports


def _write(path: Path, value: dict[str, Any]) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def _spec(raw: dict[str, Any]) -> tuple[ReplayConfig, list[Strategy], Strategy]:
    if set(raw) != {
        "execution",
        "trials",
        "benchmark",
        "open_column",
        "volume_column",
        "price_basis",
        "n_boot",
        "block_sessions",
        "seed",
    }:
        raise ValueError("tournament specification has missing or unknown keys")
    execution = ReplayConfig(**raw["execution"])
    execution.validate()

    def strategy_from_dict(item: dict[str, Any]) -> Strategy:
        fields = dict(item)
        if fields.get("allocation") is not None:
            fields["allocation"] = AllocationConfig(**fields["allocation"])
        return Strategy(**fields)

    trials = [strategy_from_dict(item) for item in raw["trials"]]
    baseline = strategy_from_dict(raw["benchmark"])
    for strategy in [baseline, *trials]:
        strategy.validate()
    if not 1 <= len(trials) <= 50 or len({s.name for s in [baseline, *trials]}) != len(trials) + 1:
        raise ValueError("1..50 uniquely named candidates and a separate benchmark required")
    for candidate in trials:
        if candidate.allocation is not None:
            if execution.funding_apr < execution.cash_apr:
                raise ValueError("cost-aware allocation requires funding_apr >= cash_apr")
            if not any(replace(candidate, name=c.name, allocation=None) == c for c in trials):
                raise ValueError(
                    "each cost-aware candidate requires an otherwise identical rank control"
                )
    if baseline.family != "equal_weight":
        raise ValueError("the benchmark must be equal_weight")
    if raw["price_basis"] not in {"raw_price_return", "consistently_adjusted_price_return"}:
        raise ValueError(
            "declare a price-return basis; total-return prices cannot be used as fills"
        )
    for name in ("open_column", "volume_column"):
        if not isinstance(raw[name], str) or not raw[name]:
            raise ValueError(f"{name} is required")
    if type(raw["n_boot"]) is not int or not 99 <= raw["n_boot"] <= 10000:
        raise ValueError("n_boot must be an integer in [99, 10000]")
    if type(raw["block_sessions"]) is not int or not 1 <= raw["block_sessions"] <= 252:
        raise ValueError("block_sessions must be an integer in [1, 252]")
    if type(raw["seed"]) is not int or raw["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    return execution, trials, baseline


def prepare_tournament(benchmark_run: Path, spec_path: Path, output: Path) -> dict[str, Any]:
    benchmark = _read_receipt(benchmark_run / "manifest.json")
    if benchmark["code_sha256"] != real_benchmark._code_sha() or benchmark["runtime"] != _runtime():
        raise ValueError("benchmark code/runtime differs from the frozen receipt")
    protocol = real_benchmark.protocol_for_run(benchmark, benchmark_run)
    if protocol.horizon_sessions != 1 or protocol.decision_delay_seconds != 0:
        raise ValueError(
            "this daily tournament requires a one-session horizon and close-time decisions"
        )
    raw = json.loads(spec_path.read_text())
    _, trials, baseline = _spec(raw)
    frame = _load_bars(protocol)
    market_panel(frame, open_column=raw["open_column"], volume_column=raw["volume_column"])
    manifest = _seal(
        {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "benchmark_manifest": benchmark,
            "benchmark_run": os.path.relpath(benchmark_run.resolve(), output.resolve()),
            "spec": raw,
            "code_sha256": _code_hashes(),
            "runtime": _tournament_runtime(),
            "candidate_count": len(trials),
            "candidates": [asdict(t) for t in trials],
            "benchmark": asdict(baseline),
            "selection_rule": "highest validation mean net return, name ascending for ties",
            "scenarios": {"configured": 1.0, "double_impact": 2.0},
            "research_only": True,
            "live_pnl_claim": False,
            "limitations": benchmark["limitations"]
            + [
                "Daily simulated price returns; consistent open/close share units must be verified upstream.",
                "Dividends, corporate actions and real borrow locates are not modeled. APRs are declared scenario inputs.",
                "Trades execute at next session open with modeled costs; opening-auction depth is not observed.",
                "Gross limits are shared; long-only and long-short books can have different factor/net exposures.",
                "Trial control covers this frozen slate, not undisclosed experiments in other run directories.",
                "Allocation return/uncertainty inputs are unvalidated proxies; pair ablations are descriptive.",
                "Impact stress changes realized costs; allocation planning coefficients stay frozen.",
            ],
        }
    )
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "manifest.json", manifest)
    return manifest


def compare_returns(
    outcomes: dict[str, Any],
    names: list[str],
    benchmark: str,
    *,
    n_boot: int,
    block: int,
    seed: int,
    min_dates: int,
) -> dict[str, Any]:
    all_names = [benchmark, *names]
    if any(outcomes[n]["status"] != "completed" for n in all_names):
        return {"status": "incomplete_trials", "tested_candidates": names}
    dates = [[row["date"] for row in outcomes[n]["daily"]] for n in all_names]
    if any(d != dates[0] for d in dates[1:]):
        raise ValueError("trial calendars do not match; rows must not be dropped")
    returns = np.array([[r["net_return"] for r in outcomes[n]["daily"]] for n in all_names]).T
    if not np.isfinite(returns).all():
        raise ValueError("non-finite trial returns")
    f = returns[:, 1:] - returns[:, :1]
    report: dict[str, Any] = {
        "tested_candidates": names,
        "n_dates": len(f),
        "mean_excess_net_return": dict(zip(names, f.mean(axis=0).tolist(), strict=True)),
    }
    # Preserve the complete slate: never silently exclude constant/failed trials.
    spread = np.ptp(f, axis=0)
    degenerate = spread <= 1e-10 * np.maximum(np.abs(f).max(axis=0), 1e-12)
    if len(f) < max(30, min_dates) or degenerate.any():
        return {
            **report,
            "status": "insufficient_or_degenerate",
            "degenerate_candidates": [n for n, bad in zip(names, degenerate, strict=True) if bad],
        }
    kwargs: dict[str, Any] = {"n_boot": n_boot, "block": float(block), "seed": seed}
    rc, spa, step = (
        snooping.reality_check(f, **kwargs),
        snooping.spa_test(f, **kwargs),
        snooping.stepm(f, **kwargs),
    )
    values = [rc.p_value, spa.p_consistent, *step.adjusted_p]
    if not np.isfinite(values).all() or spa.n_dropped or step.n_dropped or rc.n_rows_dropped:
        return {**report, "status": "untestable_full_slate"}
    return {
        **report,
        "status": "tested",
        "reality_check_p": rc.p_value,
        "spa_consistent_p": spa.p_consistent,
        "stepm_adjusted_p": dict(zip(names, step.adjusted_p, strict=True)),
        "block_sessions": block,
        "n_boot": n_boot,
        "alpha": 0.05,
    }


def run_tournament(run_dir: Path, phase: str) -> dict[str, Any]:
    if phase not in {"validation", "test"}:
        raise ValueError("phase must be validation or test")
    manifest = _read_receipt(run_dir / "manifest.json")
    if manifest["code_sha256"] != _code_hashes() or manifest["runtime"] != _tournament_runtime():
        raise ValueError("tournament code/runtime changed")
    selected = None
    validation_digest = None
    if phase == "test":
        validation = _read_receipt(run_dir / "validation.json")
        if (
            validation.get("phase") != "validation"
            or validation.get("manifest_sha256") != manifest["receipt_sha256"]
        ):
            raise ValueError("validation receipt belongs to another tournament")
        selected, validation_digest = validation["selected"], validation["receipt_sha256"]
        if selected is None:
            raise ValueError("validation was incomplete; no candidate was selected")
    raw = manifest["spec"]
    execution, trials, baseline = _spec(raw)
    benchmark_run = (run_dir.resolve() / manifest["benchmark_run"]).resolve()
    if _read_receipt(benchmark_run / "manifest.json") != manifest["benchmark_manifest"]:
        raise ValueError("benchmark manifest no longer matches the frozen tournament")
    protocol = real_benchmark.protocol_for_run(manifest["benchmark_manifest"], benchmark_run)
    panel = market_panel(
        _load_bars(protocol), open_column=raw["open_column"], volume_column=raw["volume_column"]
    )
    lo, hi = (
        (protocol.validation_start, protocol.validation_end)
        if phase == "validation"
        else (protocol.test_start, protocol.test_end)
    )
    names = [t.name for t in trials]
    # Opening this file is the phase's exclusive attempt reservation. Crashes
    # remain visible; neither a failed nor successful attempt is overwritten.
    _write(
        run_dir / f"{phase}.attempt.json",
        _seal(
            {
                "manifest_sha256": manifest["receipt_sha256"],
                "phase": phase,
                "created_at": datetime.now(UTC).isoformat(),
                "candidates": names,
            }
        ),
    )
    scenarios: dict[str, Any] = {}
    history = max(t.required_history for t in [baseline, *trials])
    for scenario, multiplier in manifest["scenarios"].items():
        outcomes = {}
        for trial in [baseline, *trials]:
            try:
                outcomes[trial.name] = replay(
                    panel,
                    trial,
                    execution,
                    start=lo,
                    end=hi,
                    history=history,
                    impact_multiplier=multiplier,
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                outcomes[trial.name] = {
                    "status": "failed",
                    "error": str(exc),
                    "trial": asdict(trial),
                }
                if isinstance(exc, AllocationFailure):
                    outcomes[trial.name]["solver_diagnostic"] = exc.diagnostic
        comparison = compare_returns(
            outcomes,
            names,
            baseline.name,
            n_boot=raw["n_boot"],
            block=raw["block_sessions"],
            seed=raw["seed"],
            min_dates=protocol.min_score_dates,
        )
        scenarios[scenario] = {
            "trials": outcomes,
            "comparison": comparison,
            "allocation_ablations": allocation_ablations(outcomes, trials),
        }
    complete = all(
        all(t["status"] == "completed" for t in case["trials"].values())
        for case in scenarios.values()
    )
    if (
        phase == "validation"
        and complete
        and scenarios["configured"]["comparison"]["status"] == "tested"
    ):
        results = scenarios["configured"]["trials"]
        selected = min(
            names,
            key=lambda name: (
                -float(np.mean([r["net_return"] for r in results[name]["daily"]])),
                name,
            ),
        )
    comparison = scenarios["configured"]["comparison"]
    selected_rejects = bool(
        phase == "test"
        and comparison["status"] == "tested"
        and selected is not None
        and comparison["stepm_adjusted_p"][selected] < 0.05
        and comparison["mean_excess_net_return"][selected] > 0
    )
    liquidated = complete and all(
        trial["liquidation_complete"]
        for case in scenarios.values()
        for trial in case["trials"].values()
    )
    stress = scenarios["double_impact"]["comparison"]
    stress_positive = bool(
        complete
        and selected is not None
        and stress.get("mean_excess_net_return", {}).get(selected, 0.0) > 0
    )
    report = _seal(
        {
            "manifest_sha256": manifest["receipt_sha256"],
            "phase": phase,
            "created_at": datetime.now(UTC).isoformat(),
            "validation_receipt_sha256": validation_digest,
            "selected": selected,
            "complete": complete,
            "scenarios": scenarios,
            "selected_holdout_adjusted_rejection": selected_rejects,
            "all_terminal_liquidations_complete": liquidated,
            "selected_survives_double_impact": stress_positive,
            "economic_evidence_gate": bool(selected_rejects and liquidated and stress_positive),
            "holdout_status": manifest["benchmark_manifest"]["holdout_status"],
            "research_only": True,
            "live_pnl_claim": False,
            "promote": False,
            "claim": "simulated_net_price_return_tournament",
            "limitations": manifest["limitations"],
        }
    )
    _write(run_dir / f"{phase}.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--benchmark-run", required=True, type=Path)
    prepare.add_argument("--spec", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    run = commands.add_parser("run")
    run.add_argument("--run", required=True, type=Path)
    run.add_argument("--phase", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    try:
        result = (
            prepare_tournament(args.benchmark_run, args.spec, args.output)
            if args.command == "prepare"
            else run_tournament(args.run, args.phase)
        )
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"scenarios", "benchmark_manifest"}},
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
