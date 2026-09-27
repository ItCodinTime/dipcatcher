"""`quant leakage` CLI tests (DESIGN.md §6.5). Mounting is W5's job (§9.2)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from quant_fund.leakage.cli import leakage_app
from quant_fund.proofcore.contracts import LeakageReport

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "leakage_fixtures"
runner = CliRunner()


def test_scan_text_exit_1_on_errors() -> None:
    result = runner.invoke(
        leakage_app, ["scan", "--paths", str(FIXTURE_DIR / "lh001_shift_forward.py")]
    )
    assert result.exit_code == 1
    assert "LH001" in result.output


def test_scan_json_format_parses_to_schema() -> None:
    result = runner.invoke(
        leakage_app,
        ["scan", "--paths", str(FIXTURE_DIR), "--format", "json", "--fail-on", "none"],
    )
    assert result.exit_code == 0
    report = LeakageReport.model_validate(json.loads(result.output))
    assert report.errors > 0
    fired = {f.rule_id for f in report.findings}
    assert {"LH001", "LH006", "LH008"} <= fired


def test_scan_clean_control_exit_0() -> None:
    result = runner.invoke(
        leakage_app, ["scan", "--paths", str(FIXTURE_DIR / "clean_fold_scaler.py")]
    )
    assert result.exit_code == 0, result.output


def test_scan_fail_on_warning() -> None:
    leaky_warning_only = FIXTURE_DIR / "lh010_bfill_event_time.py"
    result = runner.invoke(
        leakage_app, ["scan", "--paths", str(leaky_warning_only), "--fail-on", "error"]
    )
    assert result.exit_code == 0  # warning only, fail-on=error
    result = runner.invoke(
        leakage_app, ["scan", "--paths", str(leaky_warning_only), "--fail-on", "warning"]
    )
    assert result.exit_code == 1


def test_scan_rules_filter() -> None:
    result = runner.invoke(
        leakage_app,
        ["scan", "--paths", str(FIXTURE_DIR / "lh001_shift_forward.py"), "--rules", "LH002"],
    )
    assert result.exit_code == 0  # LH001 disabled -> clean


def test_scan_unknown_rule_exit_2() -> None:
    result = runner.invoke(leakage_app, ["scan", "--paths", str(FIXTURE_DIR), "--rules", "LH999"])
    assert result.exit_code == 2


def test_scan_bad_format_exit_2() -> None:
    result = runner.invoke(leakage_app, ["scan", "--paths", str(FIXTURE_DIR), "--format", "yaml"])
    assert result.exit_code == 2


def test_scan_missing_target_exit_2(tmp_path: Path) -> None:
    result = runner.invoke(leakage_app, ["scan", "--paths", str(tmp_path / "missing.py")])
    assert result.exit_code == 2
    assert "scan target" in result.output
