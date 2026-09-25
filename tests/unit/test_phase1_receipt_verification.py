"""Completed Phase-1 receipt verification using generated prices only."""

import gzip
import hashlib
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.research.catalog import BENCHMARK_CATALOG_VERSION
from quant_fund.research.net_tournament import prepare_tournament, run_tournament
from quant_fund.research.phase1_verify import verify_phase1_index, verify_phase1_run
from quant_fund.research.real_benchmark import _seal, prepare_benchmark, score_benchmark
from quant_fund.utils.reproducibility import git_revision, git_worktree_sha256


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False))


@pytest.fixture(scope="module")
def prepared_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase1_verify")
    dates = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=n) for n in range(240)]
    rng = np.random.default_rng(73)
    frames = []
    for name in ("A", "B", "C", "D"):
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, len(dates))))
        frames.append(
            pl.DataFrame(
                {
                    "security_id": [name] * len(dates),
                    "event_time": dates,
                    "available_time": dates,
                    "ingested_time": dates,
                    "source": ["test_contract"] * len(dates),
                    "close": prices,
                    "open": prices * np.exp(rng.normal(0, 0.003, len(dates))),
                    "volume": [1_000_000] * len(dates),
                }
            )
        )
    data = root / "bars.parquet"
    pl.concat(frames).write_parquet(data)
    protocol = {
        "dataset_path": "bars.parquet",
        "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "source_url": "https://example.com/test-contract",
        "usage_basis": "generated contract test",
        "price_column": "close",
        "price_adjustment": "unadjusted generated fixture",
        "universe_description": "four generated securities",
        "survivorship_bias": True,
        "availability_basis": "reconstructed",
        "holdout_previously_inspected": True,
        "train_start": "2020-01-01",
        "train_end": "2020-03-31",
        "validation_start": "2020-04-05",
        "validation_end": "2020-05-31",
        "test_start": "2020-06-05",
        "test_end": "2020-08-20",
        "min_train_rows": 100,
        "min_score_dates": 30,
    }
    config = root / "benchmark.json"
    _write(config, protocol)
    benchmark = root / "benchmark"
    prepare_benchmark(config, benchmark)
    score_benchmark(benchmark, "validation")
    score_benchmark(benchmark, "test")
    spec = {
        "execution": {},
        "trials": [
            {"name": "mom", "family": "momentum"},
            {"name": "rev", "family": "reversal", "lookback": 1},
        ],
        "benchmark": {"name": "equal", "family": "equal_weight"},
        "open_column": "open",
        "volume_column": "volume",
        "price_basis": "raw_price_return",
        "n_boot": 99,
        "block_sessions": 5,
        "seed": 17,
    }
    spec_path = root / "slate.json"
    _write(spec_path, spec)
    tournament = root / "tournament"
    prepare_tournament(benchmark, spec_path, tournament)
    run_tournament(tournament, "validation")
    run_tournament(tournament, "test")
    index_dir = root / "research"
    index_dir.mkdir()
    runs = []
    for kind, name, source in (
        ("real_benchmark", "benchmark", config),
        ("net_tournament", "tournament", spec_path),
    ):
        directory = root / name
        runs.append(
            {
                "kind": kind,
                "path": f"../{name}",
                "config_path": f"../{source.name}",
                "config_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                **{
                    f"{phase}_sha256": json.loads((directory / f"{phase}.json").read_text())[
                        "receipt_sha256"
                    ]
                    for phase in ("manifest", "validation", "test")
                },
            }
        )
    index = _seal(
        {
            "kind": "phase1_evidence_index",
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "git_revision": git_revision(),
            "git_worktree_sha256": git_worktree_sha256(),
            "benchmark_catalog_version": BENCHMARK_CATALOG_VERSION,
            "runs": runs,
        }
    )
    _write(index_dir / "index.json", index)
    return root


@pytest.fixture
def runs(prepared_runs, tmp_path):
    destination = tmp_path / "copied"
    shutil.copytree(prepared_runs, destination)
    return destination


def test_completed_runs_and_index_survive_relocation(runs):
    assert verify_phase1_run(runs / "benchmark")["valid"]
    assert verify_phase1_run(runs / "tournament")["valid"]
    index = runs / "research/index.json"
    assert verify_phase1_index(index)["valid"]
    response = CliRunner().invoke(app, ["verify-research", str(index)])
    assert response.exit_code == 0, response.output


def test_modified_dataset_and_missing_phase_fail_closed(runs):
    (runs / "bars.parquet").write_bytes((runs / "bars.parquet").read_bytes() + b"changed")
    assert any(
        "dataset SHA-256 mismatch" in e for e in verify_phase1_run(runs / "benchmark")["errors"]
    )
    (runs / "tournament/test.attempt.json").unlink()
    assert any("test.attempt.json" in e for e in verify_phase1_run(runs / "tournament")["errors"])


def test_resealed_promotion_and_broken_validation_link_fail(runs):
    report_path = runs / "benchmark/test.json"
    report = json.loads(report_path.read_text())
    report.pop("receipt_sha256")
    report["promote"] = True
    _write(report_path, _seal(report))
    assert any(
        "promote must be false" in e for e in verify_phase1_run(runs / "benchmark")["errors"]
    )
    path = runs / "tournament/test.json"
    report = json.loads(path.read_text())
    report.pop("receipt_sha256")
    report["validation_receipt_sha256"] = "0" * 64
    _write(path, _seal(report))
    assert any(
        "validation link mismatch" in e for e in verify_phase1_run(runs / "tournament")["errors"]
    )


def test_index_rejects_changed_config_and_digest_link(runs):
    (runs / "slate.json").write_text((runs / "slate.json").read_text() + " ")
    assert any(
        "config SHA-256 mismatch" in e
        for e in verify_phase1_index(runs / "research/index.json")["errors"]
    )
    path = runs / "research/index.json"
    index = json.loads(path.read_text())
    index.pop("receipt_sha256")
    index["runs"][0]["validation_sha256"] = "0" * 64
    _write(path, _seal(index))
    assert any("validation_sha256 link mismatch" in e for e in verify_phase1_index(path)["errors"])


def test_index_rejects_nonexistent_git_revision(runs):
    path = runs / "research/index.json"
    index = json.loads(path.read_text())
    index.pop("receipt_sha256")
    index["git_revision"] = "a" * 40
    _write(path, _seal(index))
    assert any("does not identify a local commit" in e for e in verify_phase1_index(path)["errors"])


def test_gzip_report_verifies_and_duplicate_representation_fails(runs):
    path = runs / "tournament/validation.json"
    original = path.read_bytes()
    with gzip.open(path.with_name(path.name + ".gz"), "wb") as handle:
        handle.write(original)
    path.unlink()
    assert verify_phase1_run(runs / "tournament")["valid"]
    assert verify_phase1_index(runs / "research/index.json")["valid"]
    path.write_bytes(original)
    assert any(
        "ambiguous raw and gzip" in e for e in verify_phase1_run(runs / "tournament")["errors"]
    )


def test_resealed_malformed_scenario_reports_errors_without_crashing(runs):
    path = runs / "tournament/validation.json"
    report = json.loads(path.read_text())
    report.pop("receipt_sha256")
    report["scenarios"]["configured"] = {"trials": None, "comparison": None}
    _write(path, _seal(report))
    result = verify_phase1_run(runs / "tournament")
    assert not result["valid"]
    assert any("trial slate incomplete" in error for error in result["errors"])


def test_blocked_validation_is_verified_without_claiming_a_holdout(runs, monkeypatch):
    import quant_fund.research.net_tournament as tournament_module

    spec_path = runs / "slate.json"
    blocked = runs / "blocked_tournament"
    prepare_tournament(runs / "benchmark", spec_path, blocked)
    original = tournament_module.replay

    def fail_candidate(panel, trial, *args, **kwargs):
        if trial.name == "mom":
            raise ValueError("injected allocator failure")
        return original(panel, trial, *args, **kwargs)

    monkeypatch.setattr(tournament_module, "replay", fail_candidate)
    validation = run_tournament(blocked, "validation")
    assert validation["selected"] is None and validation["complete"] is False
    result = verify_phase1_run(blocked)
    assert result["valid"] and result["state"] == "blocked", result["errors"]
    assert not (blocked / "test.json").exists()
    index_path = runs / "research/index.json"
    index = json.loads(index_path.read_text())
    index.pop("receipt_sha256")
    index["runs"].append(
        {
            "kind": "net_tournament",
            "path": "../blocked_tournament",
            "config_path": "../slate.json",
            "config_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            "manifest_sha256": json.loads((blocked / "manifest.json").read_text())[
                "receipt_sha256"
            ],
            "validation_sha256": validation["receipt_sha256"],
            "test_sha256": None,
        }
    )
    _write(index_path, _seal(index))
    assert verify_phase1_index(index_path)["valid"]
    (blocked / "test.attempt.json").write_text("{}")
    assert any("unexpected test.attempt.json" in e for e in verify_phase1_run(blocked)["errors"])


def test_json_path_keeps_canonical_notebook_verifier(runs):
    path = runs / "unrelated.json"
    _write(path, {"name": "not a research notebook"})
    response = CliRunner().invoke(app, ["verify-research", str(path)])
    assert response.exit_code == 1
    assert "invalid_notebook" in response.output
