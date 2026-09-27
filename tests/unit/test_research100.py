from pathlib import Path

import click
import pytest
from typer.testing import CliRunner

from quant_fund.research.research100 import (
    audit_catalog,
    describe_method,
    load_catalog,
    resolve_method,
)
from quant_fund.research.research100_cli import research100_app


@pytest.mark.parametrize("row", load_catalog(), ids=lambda r: r["id"])
def test_catalog_binds_real_components_with_tests_and_honest_scope(row):
    assert callable(resolve_method(row["id"]))
    assert row["source_url"].startswith("https://")
    assert row["test_paths"]
    root = Path(__file__).resolve().parents[2]
    assert all((root / p).is_file() for p in row["test_paths"])
    assert row["empirical_reproduction"] == "not_reproduced"
    assert row["live_performance_evidence"] is False


def test_catalog_counts_and_rejects_arbitrary_imports():
    rows = load_catalog()
    assert len(rows) == len({r["id"] for r in rows}) == len({r["title"] for r in rows}) == 100
    assert sum(r["implementation_status"] == "new_component" for r in rows) == 4
    with pytest.raises(ValueError):
        resolve_method("os:system")
    result = audit_catalog()
    assert result["available"] == result["total"] == 100
    assert result["promotable"] is False
    assert all(len(c["module_sha256"]) == 64 for c in result["components"])


def test_catalog_cli_and_explicit_synthetic_requirement():
    runner = CliRunner()
    result = runner.invoke(research100_app, ["catalog", "R008"])
    assert result.exit_code == 0
    assert describe_method("R008")["source_url"] in result.output
    result = runner.invoke(research100_app, ["benchmark"])
    assert result.exit_code != 0
    assert result.exit_code == 2
    assert "--synthetic" in click.unstyle(result.output)


def test_fixed_synthetic_benchmark_retains_every_challenger():
    from quant_fund.research.research100_benchmark import synthetic_benchmark

    result = synthetic_benchmark(seed=100)
    assert result["data_label"] == "SYNTHETIC"
    assert result["promotable"] is False
    assert result["comparison"]["n_observations"] == 60
    assert result["comparison"]["n_trials_supplied"] == 4
    assert len(result["summaries"]) == 5
    assert all(
        set(item) == {"observations", "sum_cost_fraction", "total_one_way_turnover"}
        for item in result["summaries"].values()
    )
    assert result["candidate_order"] == [
        "inverse_volatility",
        "multi_period",
        "risk_constrained_kelly",
        "volatility_managed",
    ]
    assert len(result["data_sha256"]) == 64
