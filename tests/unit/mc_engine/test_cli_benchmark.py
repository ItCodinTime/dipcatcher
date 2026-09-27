"""CLI smoke and the scaling benchmark's arithmetic."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from quant_fund.mc_engine.benchmark import scaling_benchmark
from quant_fund.mc_engine.cli import app
from quant_fund.research.catalog.constants import FORBIDDEN_RESEARCH_METRIC_KEYS

runner = CliRunner()


def _keys(document: object) -> set[str]:
    found: set[str] = set()
    if isinstance(document, dict):
        for key, value in document.items():
            found.add(str(key))
            found |= _keys(value)
    elif isinstance(document, list):
        for item in document:
            found |= _keys(item)
    return found


def test_cli_run_is_research_only_and_reproducible() -> None:
    args = [
        "run",
        "--paths",
        "200",
        "--steps",
        "4",
        "--workers",
        "1",
        "--backend",
        "serial",
        "--chunk-size",
        "50",
        "--seed",
        "3",
        "--shock-mode",
        "crude",
        "--no-progress",
    ]
    first = runner.invoke(app, args)
    second = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    document = json.loads(first.stdout)
    assert document["research_only"] is True
    assert document["live_pnl_claim"] is False
    assert document["market_evidence"] is False
    assert document["status"] == "complete"
    assert document["fingerprint"] == json.loads(second.stdout)["fingerprint"]
    assert _keys(document).isdisjoint(FORBIDDEN_RESEARCH_METRIC_KEYS)


def test_cli_help_and_interface() -> None:
    help_result = runner.invoke(app, ["--help"])
    assert help_result.exit_code == 0
    assert "Research simulation only" in help_result.stdout
    interface = runner.invoke(app, ["help-interface"])
    assert interface.exit_code == 0
    assert "USER_STREAM_ID_MIN" in interface.stdout


def test_scaling_benchmark_reports_measured_ratios() -> None:
    document = scaling_benchmark(
        n_paths=80,
        n_steps=4,
        worker_counts=[1, 2],
        repeats=1,
        seed=1,
        chunk_size=40,
        backend="serial",
    )
    assert document["fingerprints_identical_across_worker_counts"] is True
    assert document["live_pnl_claim"] is False
    rows = document["rows"]
    assert rows[0]["paths_per_s_using_min_elapsed"] == pytest.approx(
        document["n_paths"] / rows[0]["min_elapsed_s"]
    )
    assert rows[1]["measured_speedup_vs_first_row"] == pytest.approx(
        rows[0]["min_elapsed_s"] / rows[1]["min_elapsed_s"]
    )
    assert "not a claim of linear scaling" in document["limitation"]
    assert _keys(document).isdisjoint(FORBIDDEN_RESEARCH_METRIC_KEYS)


def test_cli_bench_smoke() -> None:
    result = runner.invoke(
        app,
        [
            "bench",
            "--paths",
            "40",
            "--steps",
            "2",
            "--workers",
            "1",
            "--repeats",
            "1",
            "--chunk-size",
            "20",
            "--backend",
            "serial",
        ],
    )
    assert result.exit_code == 0, result.output
    document = json.loads(result.stdout)
    assert document["rows"][0]["workers"] == 1
    assert document["rows"][0]["paths_per_s_using_min_elapsed"] > 0.0
