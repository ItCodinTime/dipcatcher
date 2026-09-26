"""Render a receipt-bound evidence page.

The page is a pure function of committed receipt bytes. Numbers are copied
from those receipts. Seal checks call ``quant_fund.research.phase1_verify``
and ``quant_fund.research.verify``; this module does not reimplement them.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Any

from quant_fund.research.catalog import FORBIDDEN_RESEARCH_METRIC_KEYS
from quant_fund.research.phase1_verify import verify_phase1_index
from quant_fund.research.real_benchmark import MODELS
from quant_fund.research.verify import verify_research_artifact
from quant_fund.utils.hashing import hash_file

_PHASE1_INDEX = Path("data/metadata/research/phase1_evidence_index.json")
_PHASE1_RUNS = (
    Path("data/metadata/real_benchmark/us_wide_20260925"),
    Path("data/metadata/net_tournament/us_wide_20260925"),
    Path("data/metadata/cost_aware_tournament/us_wide_20260925"),
)
_PHASE1_SEALED_NAMES = frozenset(
    {
        "manifest.json",
        "validation.json",
        "validation.json.gz",
        "validation.attempt.json",
        "test.json",
        "test.json.gz",
        "test.attempt.json",
    }
)
_ATTESTATION_ERROR_MARKERS = (
    "runtime differs from this environment",
    "code SHA-256 differs from this checkout",
    "code hashes differ from this checkout",
    "source dataset SHA-256 mismatch",
    "source dataset unreadable",
    "differs from indexed git revision",
    "git_revision does not identify a local commit",
    "unavailable at git_revision",
    "indexed git revision unavailable",
    "source module lacks a file",
)
_LEDGER_KEYS = frozenset({"daily", "fills", "rejections", "allocations"})
_HASH_MAP_KEYS = frozenset(
    {
        "dataset_sha256",
        "bar_files_sha256",
        "inputs_sha256",
        "input_hashes",
        "script_sha256",
        "engine_sha256",
        "allocator_sha256",
        "config_sha256",
        "manifest_sha256",
        "metrics_sha256",
        "code_sha256",
    }
)
_VERBATIM_OPEN = "<!-- verbatim-receipt-text -->"
_VERBATIM_CLOSE = "<!-- /verbatim-receipt-text -->"
_PROVENANCE_HEADERS = (
    "source",
    "file sha256",
    "embedded seal",
    "seal",
    "git revision",
    "dataset hash",
    "data class",
    "promote",
    "research_only",
    "live_pnl_claim",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def receipt_paths(root: Path) -> list[Path]:
    """Committed receipts this page is allowed to read, in stable order."""
    paths = [root / _PHASE1_INDEX]
    for folder in _PHASE1_RUNS:
        directory = root / folder
        paths.extend(directory.glob("*.json"))
        paths.extend(directory.glob("*.json.gz"))
    paths.extend((root / "receipts").glob("*.json"))
    paths.extend((root / ".dsh-24x7").glob("evidence-incumbent-vectorbt*.json"))
    unique = sorted({path.resolve() for path in paths}, key=lambda path: _rel(root, path))
    missing = [path for path in unique if not path.is_file()]
    if missing:
        rendered = ", ".join(_rel(root, path) for path in missing)
        raise FileNotFoundError(f"evidence receipts missing: {rendered}")
    return unique


def load_receipt(path: Path) -> Any:
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def content_seal_errors(errors: list[str]) -> list[str]:
    """Drop attestation results that depend on this checkout rather than receipt bytes."""
    return [
        error
        for error in errors
        if not any(marker in error for marker in _ATTESTATION_ERROR_MARKERS)
    ]


def build_report(root: Path) -> tuple[str, list[str]]:
    root = root.resolve()
    paths = receipt_paths(root)
    loaded = {path: load_receipt(path) for path in paths}
    index_path = root / _PHASE1_INDEX
    index = loaded[index_path]
    if not isinstance(index, dict):
        raise TypeError("phase1 evidence index is not an object")
    verification = verify_phase1_index(index_path)
    errors = [error for error in verification.get("errors", []) if isinstance(error, str)]
    seal_errors = content_seal_errors(errors)
    notebook_errors = _notebook_seal_errors(root, loaded)
    lines: list[str] = []
    _extend(
        lines,
        _header(),
        _verification_section(root, index_path, index, seal_errors, notebook_errors),
        _forecast_section(root, loaded, index, content_ok=not seal_errors),
        _net_section(root, loaded, index, content_ok=not seal_errors),
        _cost_section(root, loaded, index, content_ok=not seal_errors),
        _incumbent_section(
            root,
            loaded,
            "qlib incumbent parity",
            [root / "receipts" / "incumbent_bench_qlib.json"],
            incumbent_key="qlib",
        ),
        _incumbent_section(
            root,
            loaded,
            "vectorbt comparisons",
            [
                path
                for path in paths
                if path.name.startswith("evidence-incumbent-vectorbt") and path.suffix == ".json"
            ],
            incumbent_key="vectorbt",
        ),
        _dip_section(root, loaded),
        _other_section(root, loaded),
    )
    text = "\n".join(lines).rstrip() + "\n"
    return text, seal_errors + notebook_errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    root = (args.root or repo_root()).resolve()
    out = args.out or (root / "docs" / "evidence" / "index.md")
    text, seal_errors = build_report(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    if seal_errors:
        print("content seal failed:", file=sys.stderr)
        for error in seal_errors:
            print(error, file=sys.stderr)
        return 1
    return 0


def _header() -> list[str]:
    return [
        "# Receipt-bound evidence",
        "",
        "This page is generated by `scripts/build_evidence_report.py` from receipts",
        "already committed in the repository. Do not edit it by hand. Regenerate",
        "with `make evidence`.",
        "",
        "Every number below is copied from a receipt field. The generator does not",
        "rerun tournaments and does not derive new scores. A receipt field is omitted",
        "when its name tokenizes to a forbidden research-headline metric, with two",
        "exceptions: the boolean flag `live_pnl_claim` is always shown, and",
        "parity-residual fields are shown with that token replaced by `<redacted>`.",
        "Failed and blocked results stay visible. Disclaimer, limitation, and",
        "known-difference strings are copied verbatim.",
        "",
    ]


def _verification_section(
    root: Path,
    index_path: Path,
    index: dict[str, Any],
    seal_errors: list[str],
    notebook_errors: list[str],
) -> list[str]:
    lines = [
        "## Verification",
        "",
        "Phase-1 receipts are sealed with `quant_fund.research.phase1_verify.verify_phase1_index`.",
        "Notebook-shaped receipts are sealed with `quant_fund.research.verify.verify_research_artifact`.",
        "Other committed receipts in this page have no embedded seal in those verifiers;",
        "their file SHA-256 is the byte identity.",
        "",
        "The verifier also attests this checkout's interpreter, package versions,",
        "working-tree source bytes, dataset file bytes, and git object availability.",
        "Those checks are not functions of receipt bytes, so their pass or fail text",
        "is not copied into this page.",
        "",
        f"Index source: `{_rel(root, index_path)}`",
        f"Index file sha256: `{hash_file(index_path)}`",
        f"Index embedded seal: `{_embedded_seal(index)}`",
        f"Index git revision: `{index.get('git_revision', 'absent')}`",
        f"Index git worktree sha256: `{index.get('git_worktree_sha256', 'absent')}`",
        f"Benchmark catalog version: `{index.get('benchmark_catalog_version', 'absent')}`",
        "",
    ]
    if seal_errors or notebook_errors:
        lines.append("Content seals: fail")
        lines.append("")
        for error in [*seal_errors, *notebook_errors]:
            lines.append(f"- {error}")
        lines.append("")
    else:
        lines.append("Content seals: pass")
        lines.append("")
    return lines


def _forecast_section(
    root: Path, loaded: dict[Path, Any], index: dict[str, Any], *, content_ok: bool
) -> list[str]:
    run_dir = root / "data/metadata/real_benchmark/us_wide_20260925"
    manifest = _obj(loaded[run_dir / "manifest.json"])
    validation = _obj(loaded[run_dir / "validation.json"])
    test = _obj(loaded[run_dir / "test.json"])
    protocol = _obj(manifest.get("protocol"))
    audit = _obj(manifest.get("audit"))
    lines = [
        "## Phase-1 forecast baselines",
        "",
        "Sealed fixed-split forecast diagnostics on the Phase-1 US snapshot.",
        "Lower date-equal-weight MSE is the receipt's scoring direction.",
        "Dataset hash on each row is `protocol.dataset_sha256` from the benchmark",
        "manifest when the file itself does not embed one. Git revision is the",
        "index `git_revision` that binds the run.",
        "",
    ]
    lines.extend(
        _provenance_block(root, loaded, index, _phase1_files(root, run_dir), content_ok=content_ok)
    )
    lines.extend(
        [
            f"Holdout status: `{manifest.get('holdout_status', 'absent')}`",
            f"Names: `{_fmt(audit.get('n_names'))}`",
            f"Rows: `{_fmt(audit.get('rows'))}`",
            f"Late rows: `{_fmt(audit.get('late_rows'))}`",
            f"Excluded windows: `{_fmt(audit.get('excluded_windows'))}`",
            f"Source labels: `{_fmt(audit.get('source_labels'))}`",
            f"Survivorship bias: `{_fmt(protocol.get('survivorship_bias'))}`",
            f"Holdout previously inspected: `{_fmt(protocol.get('holdout_previously_inspected'))}`",
            "",
        ]
    )
    split_counts = audit.get("split_counts")
    if isinstance(split_counts, dict):
        rows = []
        for phase in ("train", "validation", "test"):
            counts = split_counts.get(phase)
            if isinstance(counts, dict):
                rows.append([phase, _fmt(counts.get("rows")), _fmt(counts.get("dates"))])
        if rows:
            lines.extend(markdown_table(["split", "rows", "dates"], rows))
            lines.append("")
    score_rows: list[list[str]] = []
    for phase, report in (("validation", validation), ("test", test)):
        scores = report.get("scores")
        if not isinstance(scores, dict):
            continue
        for model in MODELS:
            values = scores.get(model)
            if not isinstance(values, dict):
                score_rows.append([phase, model, "absent", "absent", "absent", "absent"])
                continue
            score_rows.append(
                [
                    phase,
                    model,
                    _fmt(values.get("date_equal_weight_mse")),
                    _fmt(values.get("date_equal_weight_mae")),
                    _fmt(values.get("n_rows")),
                    _fmt(values.get("n_dates")),
                ]
            )
    lines.extend(
        markdown_table(
            [
                "phase",
                "model",
                "date_equal_weight_mse",
                "date_equal_weight_mae",
                "n_rows",
                "n_dates",
            ],
            score_rows,
        )
    )
    lines.append("")
    for phase, report in (("validation", validation), ("test", test)):
        scores = report.get("scores")
        if not isinstance(scores, dict):
            continue
        zero = _obj(scores.get("zero")).get("date_equal_weight_mse")
        ridge = _obj(scores.get("ridge")).get("date_equal_weight_mse")
        lines.append(f"{phase}: {_mse_relation(zero, ridge)}")
    lines.append("")
    _emit_limitations(lines, manifest.get("limitations"), run_dir / "manifest.json", root)
    return lines


def _net_section(
    root: Path, loaded: dict[Path, Any], index: dict[str, Any], *, content_ok: bool
) -> list[str]:
    run_dir = root / "data/metadata/net_tournament/us_wide_20260925"
    lines = [
        "## Net-return tournament",
        "",
        "Frozen equal-weight, momentum-20, and reversal-1 slate. Statistics below",
        "are the sealed phase reports, including Reality Check, SPA, and StepM.",
        "`total_return` and `mean_excess_net_return` are copied from those fields.",
        "The receipt claim is `simulated_net_price_return_tournament`.",
        "",
    ]
    lines.extend(
        _provenance_block(root, loaded, index, _phase1_files(root, run_dir), content_ok=content_ok)
    )
    lines.extend(_tournament_body(root, loaded, run_dir))
    return lines


def _cost_section(
    root: Path, loaded: dict[Path, Any], index: dict[str, Any], *, content_ok: bool
) -> list[str]:
    run_dir = root / "data/metadata/cost_aware_tournament/us_wide_20260925"
    entry = _index_entry(index, "../cost_aware_tournament/us_wide_20260925")
    lines = [
        "## Cost-aware construction",
        "",
        "Status: FAILED / BLOCKED",
        "",
    ]
    reasons = _failure_reasons(_obj(loaded.get(run_dir / "validation.json.gz")))
    if reasons:
        lines.append("Recorded failure reasons:")
        lines.append("")
        for scenario, candidate, reason in reasons:
            lines.append(f"- `{scenario}` / `{candidate}`: `{reason}`")
        lines.append("")
    else:
        lines.append("Recorded failure reasons: absent")
        lines.append("")
    lines.append(
        "Index `test_sha256` for this run: "
        f"`{_fmt(None if entry is None else entry.get('test_sha256'))}`."
    )
    lines.append("No test receipt is rendered because that seal is null.")
    lines.append("")
    lines.extend(
        _provenance_block(root, loaded, index, _phase1_files(root, run_dir), content_ok=content_ok)
    )
    lines.extend(_tournament_body(root, loaded, run_dir))
    lines.extend(_ablation_table(_obj(loaded.get(run_dir / "validation.json.gz"))))
    return lines


def _incumbent_section(
    root: Path,
    loaded: dict[Path, Any],
    title: str,
    paths: list[Path],
    *,
    incumbent_key: str,
) -> list[str]:
    lines = [f"## {title}", ""]
    if title.startswith("qlib"):
        lines.extend(
            [
                "Matched-workload comparison against qlib. Terminal account levels whose",
                "field names tokenize to a forbidden research-headline metric are omitted.",
                "Parity residuals from the correctness object are copied under redacted keys.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Matched-workload comparisons against vectorbt. One block per committed",
                "evidence file. Terminal account levels whose field names tokenize to a",
                "forbidden research-headline metric are omitted. Parity residuals from",
                "the correctness object are copied under redacted keys.",
                "",
            ]
        )
    for path in paths:
        payload = _obj(loaded[path])
        lines.append(f"### `{_rel(root, path)}`")
        lines.append("")
        lines.extend(
            _provenance_block(
                root,
                loaded,
                {},
                [path],
                data_class=_data_class(payload, None),
                git_revision=_git_revision(payload, None),
            )
        )
        incumbent = _obj(payload.get("incumbent"))
        lines.append(
            f"Incumbent: `{incumbent.get('name', 'absent')}` `{incumbent.get('version', 'absent')}`"
        )
        lines.append(f"Schema: `{payload.get('schema', 'absent')}`")
        if "dipcatcher_engine" in payload:
            lines.append(f"Recorded engine: `{payload.get('dipcatcher_engine')}`")
        environment = payload.get("environment")
        if isinstance(environment, dict):
            lines.append(
                "Recorded environment: "
                + ", ".join(f"`{key}`=`{_fmt(environment[key])}`" for key in sorted(environment))
            )
        lines.append("")
        omitted: list[str] = []
        lines.extend(_correctness_table(payload.get("correctness"), omitted))
        lines.extend(_latency_table(payload.get("latency"), incumbent_key))
        lines.extend(_workload_block(payload.get("workload"), omitted))
        fault = payload.get("fault_injection")
        if fault is not None:
            lines.append("Fault injection (forbidden-key fields omitted):")
            lines.append("")
            lines.extend(render_tree(fault, omitted=omitted))
            lines.append("")
        if omitted:
            lines.append(
                f"Omitted {len(omitted)} fields whose names tokenize to a forbidden "
                "research-headline metric."
            )
            lines.append("")
        _emit_named_verbatim(lines, "Disclaimer", payload.get("disclaimer"), path, root)
        _emit_string_list(
            lines,
            "Known semantic differences",
            payload.get("known_semantic_differences"),
            path,
            root,
        )
    return lines


def _dip_section(root: Path, loaded: dict[Path, Any]) -> list[str]:
    path = root / "receipts" / "dip_bench_crypto_1d_20260925.json"
    payload = _obj(loaded[path])
    lines = [
        "## Crypto dip bench",
        "",
        "Climatology baseline on the committed crypto dip-bench receipt.",
        "Brier, ECE, and log loss are the receipt's proper scores.",
        "",
    ]
    lines.extend(
        _provenance_block(
            root,
            loaded,
            {},
            [path],
            data_class=_data_class(payload, None),
            git_revision=_git_revision(payload, None),
        )
    )
    lines.append(f"Recorded data label: `{payload.get('data', 'absent')}`")
    lines.append(f"Events: `{_fmt(payload.get('n_events'))}`")
    lines.append(f"Generated at: `{payload.get('generated_at', 'absent')}`")
    lines.append("")
    params = payload.get("params")
    if isinstance(params, dict):
        lines.append("Parameters:")
        lines.append("")
        lines.extend(render_tree(params, omitted=[]))
        lines.append("")
    recovery = payload.get("baseline_recovery")
    if isinstance(recovery, dict):
        lines.extend(
            markdown_table(
                ["horizon", "baseline_recovery"],
                [[key, _fmt(recovery[key])] for key in sorted(recovery)],
            )
        )
        lines.append("")
    forecast = _obj(payload.get("climatology_forecast"))
    definition = forecast.get("definition")
    if isinstance(definition, str):
        lines.append("Climatology definition:")
        lines.append("")
        lines.append(definition)
        lines.append("")
    metrics = forecast.get("metrics")
    if isinstance(metrics, dict):
        lines.extend(
            markdown_table(
                ["metric", "value"],
                [[key, _fmt(metrics[key])] for key in sorted(metrics)],
            )
        )
        lines.append("")
    per_asset = payload.get("per_asset")
    if isinstance(per_asset, dict):
        rows = []
        for name in sorted(per_asset):
            item = per_asset[name]
            if isinstance(item, dict):
                rows.append([name, _fmt(item.get("bars")), _fmt(item.get("events"))])
        lines.extend(markdown_table(["asset", "bars", "events"], rows))
        lines.append("")
    _emit_named_verbatim(lines, "Disclaimer", payload.get("disclaimer"), path, root)
    return lines


def _other_section(root: Path, loaded: dict[Path, Any]) -> list[str]:
    handled = {
        _rel(root, root / "receipts" / "incumbent_bench_qlib.json"),
        _rel(root, root / "receipts" / "dip_bench_crypto_1d_20260925.json"),
    }
    paths = [
        path
        for path in loaded
        if _rel(root, path).startswith("receipts/") and _rel(root, path) not in handled
    ]
    lines = [
        "## Other committed receipts",
        "",
        "Remaining files under `receipts/`. Result fields are copied. Fields whose",
        "names tokenize to a forbidden research-headline metric are omitted and",
        "counted. Prose that contains such a token is quoted verbatim.",
        "",
    ]
    for path in paths:
        payload = loaded[path]
        lines.append(f"### `{_rel(root, path)}`")
        lines.append("")
        if isinstance(payload, dict):
            lines.extend(
                _provenance_block(
                    root,
                    loaded,
                    {},
                    [path],
                    data_class=_data_class(payload, None),
                    git_revision=_git_revision(payload, None),
                )
            )
            omitted: list[str] = []
            lines.extend(render_tree(_without_provenance_hashes(payload), omitted=omitted))
            lines.append("")
            if omitted:
                lines.append(
                    f"Omitted {len(omitted)} fields whose names tokenize to a forbidden "
                    "research-headline metric."
                )
                lines.append("")
        else:
            lines.append("Receipt is not a JSON object.")
            lines.append("")
    return lines


def _tournament_body(root: Path, loaded: dict[Path, Any], run_dir: Path) -> list[str]:
    lines: list[str] = []
    reports: list[tuple[str, dict[str, Any]]] = []
    missing: list[str] = []
    for phase in ("validation", "test"):
        path = _phase_report_path(run_dir, phase)
        if path is None:
            missing.append(phase)
            continue
        report = _obj(loaded[path])
        reports.append((phase, report))
    if reports:
        lines.extend(
            markdown_table(
                [
                    "phase",
                    "selected",
                    "complete",
                    "economic_evidence_gate",
                    "selected_holdout_adjusted_rejection",
                    "selected_survives_double_impact",
                    "all_terminal_liquidations_complete",
                    "promote",
                    "live_pnl_claim",
                ],
                [
                    [
                        phase,
                        _fmt(report.get("selected")),
                        _fmt(report.get("complete")),
                        _fmt(report.get("economic_evidence_gate")),
                        _fmt(report.get("selected_holdout_adjusted_rejection")),
                        _fmt(report.get("selected_survives_double_impact")),
                        _fmt(report.get("all_terminal_liquidations_complete")),
                        _fmt(report.get("promote")),
                        _fmt(report.get("live_pnl_claim")),
                    ]
                    for phase, report in reports
                ],
            )
        )
        lines.append("")
    trial_rows: list[list[str]] = []
    comparison_rows: list[list[str]] = []
    diagnostic_rows: list[list[str]] = []
    for phase, report in reports:
        scenarios = report.get("scenarios")
        if not isinstance(scenarios, dict):
            continue
        for scenario, case in scenarios.items():
            if not isinstance(case, dict):
                continue
            trials = case.get("trials")
            if isinstance(trials, dict):
                for name, trial in trials.items():
                    trial_obj = _obj(trial)
                    trial_rows.append(
                        [
                            phase,
                            str(scenario),
                            str(name),
                            _fmt(trial_obj.get("status")),
                            _fmt(trial_obj.get("total_return")),
                            _fmt(trial_obj.get("rejected_orders")),
                            _fmt(trial_obj.get("max_drawdown")),
                            _fmt(trial_obj.get("liquidation_complete")),
                            _fmt(trial_obj.get("error")),
                        ]
                    )
                    diagnostic = trial_obj.get("solver_diagnostic")
                    if isinstance(diagnostic, dict):
                        diagnostic_rows.append(
                            [
                                phase,
                                str(scenario),
                                str(name),
                                _fmt(diagnostic.get("solver")),
                                _fmt(diagnostic.get("status")),
                                _fmt(diagnostic.get("iterations")),
                                _fmt(diagnostic.get("solve_time_seconds")),
                                _fmt(diagnostic.get("weights_accepted")),
                            ]
                        )
            comparison = _obj(case.get("comparison"))
            excess = comparison.get("mean_excess_net_return")
            stepm = comparison.get("stepm_adjusted_p")
            names = _comparison_names(comparison, excess, stepm)
            if not names:
                comparison_rows.append(
                    [
                        phase,
                        str(scenario),
                        _fmt(comparison.get("status")),
                        "absent",
                        _fmt(comparison.get("n_dates")),
                        _fmt(comparison.get("reality_check_p")),
                        _fmt(comparison.get("spa_consistent_p")),
                        "absent",
                        "absent",
                    ]
                )
                continue
            for name in names:
                comparison_rows.append(
                    [
                        phase,
                        str(scenario),
                        _fmt(comparison.get("status")),
                        name,
                        _fmt(comparison.get("n_dates")),
                        _fmt(comparison.get("reality_check_p")),
                        _fmt(comparison.get("spa_consistent_p")),
                        _fmt(_obj(stepm).get(name) if isinstance(stepm, dict) else None),
                        _fmt(_obj(excess).get(name) if isinstance(excess, dict) else None),
                    ]
                )
    if trial_rows:
        lines.extend(
            markdown_table(
                [
                    "phase",
                    "scenario",
                    "candidate",
                    "status",
                    "total_return",
                    "rejected_orders",
                    "max_drawdown",
                    "liquidation_complete",
                    "error",
                ],
                trial_rows,
            )
        )
        lines.append("")
    if comparison_rows:
        lines.extend(
            markdown_table(
                [
                    "phase",
                    "scenario",
                    "comparison_status",
                    "candidate",
                    "n_dates",
                    "reality_check_p",
                    "spa_consistent_p",
                    "stepm_adjusted_p",
                    "mean_excess_net_return",
                ],
                comparison_rows,
            )
        )
        lines.append("")
    if diagnostic_rows:
        lines.append("Solver diagnostics copied from failed trials:")
        lines.append("")
        lines.extend(
            markdown_table(
                [
                    "phase",
                    "scenario",
                    "candidate",
                    "solver",
                    "status",
                    "iterations",
                    "solve_time_seconds",
                    "weights_accepted",
                ],
                diagnostic_rows,
            )
        )
        lines.append("")
    if missing:
        lines.append("Absent phase reports: " + ", ".join(f"`{phase}`" for phase in missing) + ".")
        lines.append("")
    limitation_source = _phase_report_path(run_dir, "validation") or (run_dir / "manifest.json")
    limitations = _obj(loaded.get(limitation_source)).get("limitations")
    if limitations is None:
        limitations = _obj(loaded[run_dir / "manifest.json"]).get("limitations")
        limitation_source = run_dir / "manifest.json"
    _emit_limitations(lines, limitations, limitation_source, root)
    return lines


def _ablation_table(report: dict[str, Any]) -> list[str]:
    scenarios = report.get("scenarios")
    if not isinstance(scenarios, dict):
        return []
    rows: list[list[str]] = []
    for scenario, case in scenarios.items():
        if not isinstance(case, dict):
            continue
        ablations = case.get("allocation_ablations")
        if not isinstance(ablations, list):
            continue
        for item in ablations:
            if not isinstance(item, dict):
                continue
            rows.append(
                [
                    str(scenario),
                    _fmt(item.get("candidate")),
                    _fmt(item.get("control")),
                    _fmt(item.get("status")),
                    _fmt(item.get("inference")),
                ]
            )
    if not rows:
        return []
    return [
        *markdown_table(
            ["scenario", "candidate", "control", "status", "inference"],
            rows,
        ),
        "",
    ]


def _correctness_table(correctness: object, omitted: list[str]) -> list[str]:
    if not isinstance(correctness, dict):
        return ["Correctness object: absent", ""]
    rows: list[list[str]] = []
    for key, value in correctness.items():
        if _key_forbidden(str(key)):
            if _is_parity_residual(str(key)) and not isinstance(value, (dict, list)):
                rows.append([redact_key(str(key)), _fmt(value)])
            else:
                omitted.append(str(key))
            continue
        if isinstance(value, list) and any(
            isinstance(item, str) and contains_forbidden_token(item) for item in value
        ):
            omitted.append(str(key))
            continue
        if isinstance(value, (dict, list)):
            continue
        rows.append([str(key), _fmt(value)])
    if not rows:
        return []
    return [*markdown_table(["field", "value"], rows), ""]


def _latency_table(latency: object, incumbent_key: str) -> list[str]:
    if not isinstance(latency, dict):
        return []
    rows: list[list[str]] = []
    preferred = [
        "reps",
        "dipcatcher_ms_median",
        f"{incumbent_key}_ms_median",
        "dipcatcher_ms_all",
        f"{incumbent_key}_ms_all",
    ]
    keys = [key for key in preferred if key in latency]
    keys.extend(key for key in latency if key not in keys)
    for key in keys:
        value = latency[key]
        if isinstance(value, list):
            rendered = ", ".join(_fmt(item) for item in value)
        else:
            rendered = _fmt(value)
        rows.append([str(key), rendered])
    return [*markdown_table(["latency field", "value"], rows), ""]


def _workload_block(workload: object, omitted: list[str]) -> list[str]:
    if not isinstance(workload, dict):
        return []
    lines = ["Workload fields copied from the receipt:", ""]
    kept = {
        key: value
        for key, value in workload.items()
        if key not in _HASH_MAP_KEYS and key not in {"bar_files_sha256"}
    }
    lines.extend(render_tree(kept, omitted=omitted))
    lines.append("")
    return lines


def _provenance_block(
    root: Path,
    loaded: dict[Path, Any],
    index: dict[str, Any],
    paths: list[Path],
    *,
    data_class: str | None = None,
    git_revision: str | None = None,
    content_ok: bool = True,
) -> list[str]:
    inherited_git = git_revision if git_revision is not None else _index_git(index)
    rows: list[list[str]] = []
    for path in paths:
        payload = loaded.get(path)
        payload_obj = payload if isinstance(payload, dict) else {}
        inherited_class = data_class
        inherited_dataset: list[tuple[str, str]] = []
        if inherited_class is None or git_revision is None:
            run_class, run_git, run_dataset = _phase1_context(root, path, loaded, index)
            if inherited_class is None:
                inherited_class = run_class
            if git_revision is None:
                inherited_git = run_git
            inherited_dataset = run_dataset
        rows.append(
            [
                _rel(root, path),
                hash_file(path),
                _embedded_seal(payload_obj),
                _seal_label(root, path, payload_obj, content_ok=content_ok),
                _git_revision(payload_obj, inherited_git),
                _format_dataset(_dataset_bindings(payload_obj, inherited_dataset)),
                _data_class(payload_obj, inherited_class),
                _flag(payload_obj, "promote"),
                _flag(payload_obj, "research_only"),
                _flag(payload_obj, "live_pnl_claim"),
            ]
        )
    return [*markdown_table(list(_PROVENANCE_HEADERS), rows), ""]


def render_tree(value: Any, *, omitted: list[str], depth: int = 0) -> list[str]:
    lines: list[str] = []
    indent = "  " * depth
    if isinstance(value, dict):
        for key, child in value.items():
            name = str(key)
            if name in _LEDGER_KEYS or name == "live_pnl_claim":
                continue
            if _key_forbidden(name):
                if _is_parity_residual(name) and not isinstance(child, (dict, list)):
                    lines.append(f"{indent}- `{redact_key(name)}`: {_fmt(child)}")
                else:
                    omitted.append(name)
                continue
            if isinstance(child, (dict, list)):
                lines.append(f"{indent}- `{name}`:")
                lines.extend(render_tree(child, omitted=omitted, depth=depth + 1))
                continue
            lines.extend(_render_scalar(name, child, indent))
        return lines
    if isinstance(value, list):
        if value and all(not isinstance(item, (dict, list)) for item in value):
            for item in value:
                if isinstance(item, str) and contains_forbidden_token(item):
                    if len(item) > 24:
                        lines.append(verbatim(item))
                    else:
                        omitted.append(item)
                else:
                    lines.append(f"{indent}- {_fmt(item) if not isinstance(item, str) else item}")
            return lines
        for index, item in enumerate(value):
            lines.append(f"{indent}- [{index}]")
            lines.extend(render_tree(item, omitted=omitted, depth=depth + 1))
        return lines
    lines.append(f"{indent}- {_fmt(value) if not isinstance(value, str) else value}")
    return lines


def verbatim(text: str) -> str:
    body = text.replace("\r\n", "\n").replace("\r", "\n")
    quoted = "\n".join(">" if line == "" else f"> {line}" for line in body.split("\n"))
    return f"{_VERBATIM_OPEN}\n{quoted}\n{_VERBATIM_CLOSE}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    def esc(cell: str) -> str:
        return cell.replace("|", "\\|").replace("\n", " ")

    head = "| " + " | ".join(esc(header) for header in headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(esc(cell) for cell in row) + " |" for row in rows]
    return [head, sep, *body]


def contains_forbidden_token(text: str) -> bool:
    return any(
        token.lower() in FORBIDDEN_RESEARCH_METRIC_KEYS
        for token in re.split(r"[^A-Za-z]+", text)
        if token
    )


def redact_key(key: str) -> str:
    parts = []
    for part in str(key).split("_"):
        if part.lower() in FORBIDDEN_RESEARCH_METRIC_KEYS:
            parts.append("<redacted>")
        else:
            parts.append(part)
    return "_".join(parts)


def _render_scalar(key: str, value: Any, indent: str) -> list[str]:
    if isinstance(value, str) and contains_forbidden_token(value):
        return [f"{indent}- `{key}`:", verbatim(value)]
    shown = value if isinstance(value, str) else _fmt(value)
    return [f"{indent}- `{key}`: {shown}"]


def _mse_relation(zero: object, ridge: object) -> str:
    if not isinstance(zero, (int, float)) or not isinstance(ridge, (int, float)):
        return "zero and ridge date-equal-weight MSE are not both numeric in the receipt."
    if isinstance(zero, bool) or isinstance(ridge, bool):
        return "zero and ridge date-equal-weight MSE are not both numeric in the receipt."
    if ridge > zero:
        return "ridge date-equal-weight MSE is higher than the zero baseline."
    if ridge < zero:
        return "ridge date-equal-weight MSE is lower than the zero baseline."
    return "ridge date-equal-weight MSE equals the zero baseline."


def _failure_reasons(report: dict[str, Any]) -> list[tuple[str, str, str]]:
    reasons: list[tuple[str, str, str]] = []
    scenarios = report.get("scenarios")
    if not isinstance(scenarios, dict):
        return reasons
    for scenario, case in scenarios.items():
        if not isinstance(case, dict) or not isinstance(case.get("trials"), dict):
            continue
        for name, trial in case["trials"].items():
            if isinstance(trial, dict) and isinstance(trial.get("error"), str):
                reasons.append((str(scenario), str(name), trial["error"]))
    return reasons


def _comparison_names(comparison: dict[str, Any], excess: object, stepm: object) -> list[str]:
    if isinstance(excess, dict) and excess:
        return [str(name) for name in excess]
    if isinstance(stepm, dict) and stepm:
        return [str(name) for name in stepm]
    tested = comparison.get("tested_candidates")
    if isinstance(tested, list):
        return [str(name) for name in tested]
    return []


def _emit_limitations(lines: list[str], limitations: object, source: Path, root: Path) -> None:
    lines.append(f"Limitations copied verbatim from `{_rel(root, source)}`:")
    lines.append("")
    if not isinstance(limitations, list) or not limitations:
        lines.append("Limitations field: absent")
        lines.append("")
        return
    for item in limitations:
        if isinstance(item, str):
            lines.append(verbatim(item))
            lines.append("")


def _emit_named_verbatim(
    lines: list[str], title: str, value: object, source: Path, root: Path
) -> None:
    lines.append(f"{title} copied verbatim from `{_rel(root, source)}`:")
    lines.append("")
    if not isinstance(value, str) or not value:
        lines.append(f"{title}: absent")
        lines.append("")
        return
    lines.append(verbatim(value))
    lines.append("")


def _emit_string_list(
    lines: list[str], title: str, value: object, source: Path, root: Path
) -> None:
    lines.append(f"{title} copied verbatim from `{_rel(root, source)}`:")
    lines.append("")
    if not isinstance(value, list) or not value:
        lines.append(f"{title}: absent")
        lines.append("")
        return
    for item in value:
        if isinstance(item, str):
            lines.append(verbatim(item))
            lines.append("")


def _notebook_seal_errors(root: Path, loaded: dict[Path, Any]) -> list[str]:
    errors: list[str] = []
    for path, payload in loaded.items():
        if not isinstance(payload, dict) or not _is_research_notebook(payload):
            continue
        result = verify_research_artifact(path)
        for error in result.get("errors", []):
            if isinstance(error, str):
                errors.append(f"{_rel(root, path)}: {error}")
    return errors


def _is_research_notebook(payload: dict[str, Any]) -> bool:
    return payload.get("firm") == "Artificial Hedge" and "rankers" in payload


def _phase1_files(root: Path, run_dir: Path) -> list[Path]:
    files = [path for path in receipt_paths(root) if path.parent == run_dir.resolve()]
    return sorted(files, key=lambda path: path.name)


def _phase_report_path(run_dir: Path, phase: str) -> Path | None:
    plain = run_dir / f"{phase}.json"
    compressed = run_dir / f"{phase}.json.gz"
    if compressed.is_file():
        return compressed
    if plain.is_file():
        return plain
    return None


def _phase1_context(
    root: Path, path: Path, loaded: dict[Path, Any], index: dict[str, Any]
) -> tuple[str | None, str | None, list[tuple[str, str]]]:
    run_dir = path.parent
    if path == (root / _PHASE1_INDEX).resolve() or run_dir not in {
        (root / folder).resolve() for folder in _PHASE1_RUNS
    }:
        if path == (root / _PHASE1_INDEX).resolve():
            return None, _index_git(index), []
        return None, None, []
    manifest_path = run_dir / "manifest.json"
    manifest = _obj(loaded.get(manifest_path))
    dataset = _dataset_bindings(manifest, [])
    if not dataset:
        embedded = manifest.get("benchmark_manifest")
        dataset = _dataset_bindings(_obj(embedded), [])
    data_class = _data_class(manifest, None)
    if data_class == "unspecified":
        embedded = _obj(manifest.get("benchmark_manifest"))
        data_class = _data_class(embedded, None)
    return data_class, _index_git(index), dataset


def _index_entry(index: dict[str, Any], suffix: str) -> dict[str, Any] | None:
    runs = index.get("runs")
    if not isinstance(runs, list):
        return None
    for entry in runs:
        if isinstance(entry, dict) and entry.get("path") == suffix:
            return entry
    return None


def _index_git(index: dict[str, Any]) -> str | None:
    value = index.get("git_revision")
    if isinstance(value, str) and value:
        return value
    return None


def _dataset_bindings(
    payload: dict[str, Any], inherited: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []

    def add_map(prefix: str, value: object) -> None:
        if isinstance(value, str) and _is_sha256(value):
            found.append((prefix, value))
        elif isinstance(value, dict):
            for name in sorted(value):
                item = value[name]
                if isinstance(item, str) and _is_sha256(item):
                    found.append((f"{prefix}.{name}", item))

    protocol = payload.get("protocol")
    if isinstance(protocol, dict):
        add_map("protocol.dataset_sha256", protocol.get("dataset_sha256"))
    embedded = payload.get("benchmark_manifest")
    if isinstance(embedded, dict) and isinstance(embedded.get("protocol"), dict):
        add_map(
            "benchmark_manifest.protocol.dataset_sha256",
            embedded["protocol"].get("dataset_sha256"),
        )
    for key in ("bar_files_sha256", "inputs_sha256", "input_hashes"):
        if key in payload:
            add_map(key, payload.get(key))
        workload = payload.get("workload")
        if isinstance(workload, dict) and key in workload:
            add_map(f"workload.{key}", workload.get(key))
    if found:
        return found
    return list(inherited)


def _format_dataset(bindings: list[tuple[str, str]]) -> str:
    if not bindings:
        return "absent"
    if len(bindings) == 1 and bindings[0][0].endswith("dataset_sha256"):
        return bindings[0][1]
    return ", ".join(f"{field}={digest}" for field, digest in bindings)


def _data_class(payload: dict[str, Any], inherited: str | None) -> str:
    if payload.get("synthetic") is True:
        return "SYNTHETIC"
    source = payload.get("data_source")
    if isinstance(source, str) and source.strip().upper() == "SYNTHETIC":
        return "SYNTHETIC"
    data = payload.get("data")
    if isinstance(data, str) and re.search(r"\breal\b", data, re.IGNORECASE):
        return "real"
    disclaimer = payload.get("disclaimer")
    if isinstance(disclaimer, str) and re.search(r"\breal\b", disclaimer, re.IGNORECASE):
        return "real"
    labels = _source_labels(payload)
    if labels and all(label.upper() != "SYNTHETIC" for label in labels):
        return "real"
    if payload.get("synthetic") is False:
        return "real"
    if inherited in {"real", "SYNTHETIC", "unspecified"}:
        return inherited
    return "unspecified"


def _source_labels(payload: dict[str, Any]) -> list[str]:
    audit = payload.get("audit")
    if isinstance(audit, dict) and isinstance(audit.get("source_labels"), list):
        return [str(label) for label in audit["source_labels"]]
    embedded = payload.get("benchmark_manifest")
    if isinstance(embedded, dict):
        return _source_labels(embedded)
    return []


def _git_revision(payload: dict[str, Any], inherited: str | None) -> str:
    value = payload.get("git_revision")
    if isinstance(value, str) and value:
        return value
    if inherited:
        return inherited
    return "absent"


def _flag(payload: dict[str, Any], key: str) -> str:
    if key not in payload:
        return "absent"
    return _fmt(payload.get(key))


def _embedded_seal(payload: dict[str, Any]) -> str:
    value = payload.get("receipt_sha256")
    if isinstance(value, str) and value:
        return value
    return "absent"


def _seal_label(root: Path, path: Path, payload: dict[str, Any], *, content_ok: bool) -> str:
    if _is_research_notebook(payload):
        result = verify_research_artifact(path)
        errors = [error for error in result.get("errors", []) if isinstance(error, str)]
        return "pass" if not errors else "fail"
    relative_parent = path.parent.resolve()
    phase1_parents = {(root / folder).resolve() for folder in _PHASE1_RUNS}
    if path.name in _PHASE1_SEALED_NAMES and relative_parent in phase1_parents:
        return "pass" if content_ok else "fail"
    if path.resolve() == (root / _PHASE1_INDEX).resolve():
        return "pass" if content_ok else "fail"
    if isinstance(payload.get("receipt_sha256"), str):
        return "not_checked"
    return "no_embedded_seal"


def _without_provenance_hashes(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in _HASH_MAP_KEYS}


def _key_forbidden(key: str) -> bool:
    parts = [part for part in key.lower().replace("-", "_").split("_") if part]
    return any(part in FORBIDDEN_RESEARCH_METRIC_KEYS for part in parts)


def _is_parity_residual(key: str) -> bool:
    redacted = redact_key(key)
    return redacted.endswith("_max_abs_diff") or redacted.endswith("_max_rel_diff")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value, allow_nan=False)
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(not isinstance(item, (dict, list)) for item in value):
        return ", ".join(_fmt(item) if not isinstance(item, str) else item for item in value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _obj(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _rel(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _extend(lines: list[str], *blocks: list[str]) -> None:
    for block in blocks:
        lines.extend(block)


if __name__ == "__main__":
    raise SystemExit(main())
