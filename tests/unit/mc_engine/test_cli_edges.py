"""CLI edge paths: argument validation, vol-target strategy, and the
checkpoint/resume round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.mc_engine.cli import app

pytestmark = pytest.mark.synthetic

runner = CliRunner()

_RUN_BASE = [
    "run",
    "--paths",
    "64",
    "--steps",
    "4",
    "--workers",
    "1",
    "--backend",
    "serial",
    "--chunk-size",
    "32",
    "--no-progress",
]


def test_floats_rejects_blank_list() -> None:
    result = runner.invoke(app, [*_RUN_BASE, "--mu", "  , "])
    assert result.exit_code != 0


def test_mu_vol_length_mismatch() -> None:
    result = runner.invoke(app, [*_RUN_BASE, "--mu", "0.01,0.02", "--vol", "0.2"])
    assert result.exit_code != 0


def test_vol_target_requires_single_asset_and_runs() -> None:
    bad = runner.invoke(
        app,
        [
            *_RUN_BASE,
            "--strategy",
            "vol-target",
            "--mu",
            "0.01,0.02",
            "--vol",
            "0.2,0.2",
        ],
    )
    assert bad.exit_code != 0
    good = runner.invoke(
        app,
        [
            *_RUN_BASE,
            "--strategy",
            "vol-target",
            "--mu",
            "0.01",
            "--vol",
            "0.2",
            "--target-vol",
            "0.15",
            "--lookback",
            "3",
        ],
    )
    assert good.exit_code == 0, good.output
    assert json.loads(good.stdout)["status"] == "complete"


def test_corr_must_be_inside_open_interval() -> None:
    result = runner.invoke(
        app,
        [*_RUN_BASE, "--mu", "0,0", "--vol", "0.2,0.2", "--corr", "1.0"],
    )
    assert result.exit_code != 0
    ok = runner.invoke(
        app,
        [
            *_RUN_BASE,
            "--mu",
            "0,0",
            "--vol",
            "0.2,0.3",
            "--corr",
            "0.4",
            "--weights",
            "0.3,0.7",
        ],
    )
    assert ok.exit_code == 0, ok.output


def test_unknown_strategy_rejected() -> None:
    result = runner.invoke(app, [*_RUN_BASE, "--strategy", "bogus"])
    assert result.exit_code != 0


def test_risk_knobs_flow_through() -> None:
    result = runner.invoke(
        app,
        [
            *_RUN_BASE,
            "--importance-shift",
            "0.1",
            "--control-variate",
            "--scrambles",
            "1",
            "--evt-threshold",
            "0.08",
            "--ruin-level",
            "0.4",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "complete"


def test_resume_rejects_dir_without_manifest(tmp_path: Path) -> None:
    result = runner.invoke(app, ["resume", "--checkpoint", str(tmp_path)])
    assert result.exit_code != 0


def test_resume_rejects_manifest_without_spec(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"mc_engine_version": 1}))
    result = runner.invoke(app, ["resume", "--checkpoint", str(tmp_path)])
    assert result.exit_code != 0


def test_run_checkpoint_then_resume_round_trip(tmp_path: Path) -> None:
    run = runner.invoke(app, [*_RUN_BASE, "--checkpoint", str(tmp_path)])
    assert run.exit_code == 0, run.output
    assert (tmp_path / "manifest.json").is_file()
    resume = runner.invoke(
        app,
        ["resume", "--checkpoint", str(tmp_path), "--backend", "serial"],
    )
    assert resume.exit_code == 0, resume.output
    document = json.loads(resume.stdout)
    assert document["status"] == "complete"
    assert document["fingerprint"] == json.loads(run.stdout)["fingerprint"]
