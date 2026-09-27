"""Unit tests for the CI baseline differ (scripts/security_baseline_diff.py)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "security_baseline_diff",
    Path(__file__).resolve().parents[3] / "scripts" / "security_baseline_diff.py",
)
assert _SPEC is not None and _SPEC.loader is not None
differ = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(differ)


def _bandit_report(results: list[dict]) -> dict:
    return {"errors": [], "metrics": {}, "results": results}


def _finding(code: str = "subprocess.call(x)", test_id: str = "B603") -> dict:
    return {
        "filename": "pkg/mod.py",
        "line_number": 12,
        "test_id": test_id,
        "test_name": "subprocess_popen",
        "issue_severity": "LOW",
        "issue_confidence": "HIGH",
        "issue_text": "subprocess call",
        "code": code,
    }


def test_new_finding_fails(tmp_path: Path) -> None:
    report = tmp_path / "r.json"
    report.write_text(json.dumps(_bandit_report([_finding()])), encoding="utf-8")
    baseline = tmp_path / "b.json"
    baseline.write_text(
        json.dumps({"schema": "security-baseline.v1", "findings": []}), encoding="utf-8"
    )
    assert (
        differ.main(["--tool", "bandit", "--report", str(report), "--baseline", str(baseline)]) == 1
    )


def test_known_finding_passes(tmp_path: Path) -> None:
    report = tmp_path / "r.json"
    report.write_text(json.dumps(_bandit_report([_finding()])), encoding="utf-8")
    baseline = tmp_path / "b.json"
    assert (
        differ.main(
            [
                "--tool",
                "bandit",
                "--report",
                str(report),
                "--baseline",
                str(baseline),
                "--update",
            ]
        )
        == 0
    )
    assert (
        differ.main(["--tool", "bandit", "--report", str(report), "--baseline", str(baseline)]) == 0
    )


def test_line_number_drift_is_not_new(tmp_path: Path) -> None:
    report = tmp_path / "r.json"
    report.write_text(json.dumps(_bandit_report([_finding()])), encoding="utf-8")
    baseline = tmp_path / "b.json"
    differ.main(
        ["--tool", "bandit", "--report", str(report), "--baseline", str(baseline), "--update"]
    )
    moved = _finding()
    moved["line_number"] = 987
    report.write_text(json.dumps(_bandit_report([moved])), encoding="utf-8")
    assert (
        differ.main(["--tool", "bandit", "--report", str(report), "--baseline", str(baseline)]) == 0
    )


def test_code_change_reflags(tmp_path: Path) -> None:
    report = tmp_path / "r.json"
    report.write_text(json.dumps(_bandit_report([_finding()])), encoding="utf-8")
    baseline = tmp_path / "b.json"
    differ.main(
        ["--tool", "bandit", "--report", str(report), "--baseline", str(baseline), "--update"]
    )
    changed = _finding(code="subprocess.call(y, shell=True)")
    report.write_text(json.dumps(_bandit_report([changed])), encoding="utf-8")
    assert (
        differ.main(["--tool", "bandit", "--report", str(report), "--baseline", str(baseline)]) == 1
    )


def test_missing_baseline_means_everything_new(tmp_path: Path) -> None:
    report = tmp_path / "r.json"
    report.write_text(json.dumps(_bandit_report([_finding()])), encoding="utf-8")
    assert (
        differ.main(
            ["--tool", "bandit", "--report", str(report), "--baseline", str(tmp_path / "nope.json")]
        )
        == 1
    )


def test_malformed_baseline_is_hard_error(tmp_path: Path) -> None:
    report = tmp_path / "r.json"
    report.write_text(json.dumps(_bandit_report([])), encoding="utf-8")
    baseline = tmp_path / "b.json"
    baseline.write_text(json.dumps({"schema": "other"}), encoding="utf-8")
    assert (
        differ.main(["--tool", "bandit", "--report", str(report), "--baseline", str(baseline)]) == 2
    )


def test_pip_audit_fingerprints(tmp_path: Path) -> None:
    report_payload = {
        "dependencies": [
            {
                "name": "requests",
                "version": "2.0.0",
                "vulns": [{"id": "GHSA-xxx", "fix_versions": ["2.1.0"], "aliases": []}],
            }
        ],
        "fixes": [],
    }
    report = tmp_path / "r.json"
    report.write_text(json.dumps(report_payload), encoding="utf-8")
    baseline = tmp_path / "b.json"
    assert (
        differ.main(["--tool", "pip-audit", "--report", str(report), "--baseline", str(baseline)])
        == 1
    )
    differ.main(
        [
            "--tool",
            "pip-audit",
            "--report",
            str(report),
            "--baseline",
            str(baseline),
            "--update",
        ]
    )
    assert (
        differ.main(["--tool", "pip-audit", "--report", str(report), "--baseline", str(baseline)])
        == 0
    )
