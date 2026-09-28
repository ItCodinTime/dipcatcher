"""P7.3: fleet tournaments indexed in MLflow, keyed to sealed receipt digests.

All data here is SYNTHETIC — correctness evidence, never market evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.registry.mlflow_store import log_fleet_run
from quant_fund.research.fleet_eval import (
    run_distribution_fleet,
    write_fleet_receipt,
)

TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)


def _tiny_fleet(tmp_path: Path, seed: int = 0):
    from quant_fund.models.distribution import (
        EmpiricalDistribution,
        GaussianDistribution,
    )

    factories = {
        "empirical": lambda: EmpiricalDistribution(list(TAUS)),
        "gaussian": lambda: GaussianDistribution(list(TAUS)),
    }
    frame, receipt = run_distribution_fleet(
        factories, ["iid_gaussian"], n_train=96, n_eval=48, seed=seed, taus=TAUS
    )
    path = write_fleet_receipt(receipt, tmp_path / "receipts")
    return frame, receipt, path


def test_log_fleet_run_indexes_receipt_digest(tmp_path: Path) -> None:
    _, receipt, path = _tiny_fleet(tmp_path)
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    run_id = log_fleet_run(path, receipt, uri=uri)
    assert isinstance(run_id, str) and run_id

    from mlflow.tracking import MlflowClient

    run = MlflowClient(tracking_uri=uri).get_run(run_id)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert run.data.tags["receipt_sha256"] == digest
    assert run.data.tags["data"] == "SYNTHETIC"
    assert run.data.tags["evidence"] == "receipt"
    assert run.data.params["receipt_path"] == str(path)
    metrics = run.data.metrics
    assert metrics["n_heads"] == 2.0 and metrics["n_shards"] == 1.0
    assert metrics["n_ok_rows"] == 2.0 and metrics["n_error_rows"] == 0.0
    assert metrics["mean_crps"] > 0.0
    assert metrics["best_head_crps"] <= metrics["worst_head_crps"]
    assert metrics["mean_pit_ks"] >= 0.0


def test_log_fleet_run_fail_closed(tmp_path: Path) -> None:
    _, receipt, path = _tiny_fleet(tmp_path)
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    with pytest.raises(ValueError, match="missing or not a regular file"):
        log_fleet_run(tmp_path / "nope.json", receipt, uri=uri)
    with pytest.raises(ValueError, match="data_label"):
        log_fleet_run(path, {**receipt, "data_label": ""}, uri=uri)
    with pytest.raises(ValueError, match="results"):
        log_fleet_run(path, {**receipt, "results": "nope"}, uri=uri)
    all_err = {
        **receipt,
        "results": [{**r, "status": "error"} for r in receipt["results"]],
    }
    with pytest.raises(ValueError, match="no successfully scored rows"):
        log_fleet_run(path, all_err, uri=uri)


def test_cli_fleet_mlflow_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`fleet --mlflow` echoes a run id and writes the index row."""
    monkeypatch.chdir(tmp_path)
    config = Path(__file__).resolve().parents[3] / "configs" / "research.yaml"
    result = CliRunner().invoke(
        app,
        [
            "fleet",
            "--config",
            str(config),
            "--models",
            "empirical,gaussian",
            "--shards",
            "iid_gaussian",
            "--n-train",
            "96",
            "--n-eval",
            "48",
            "--out-dir",
            str(tmp_path / "receipts"),
            "--mlflow",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "receipt=" in result.output
    run_line = next(
        line for line in result.output.splitlines() if line.startswith("mlflow_run_id=")
    )
    run_id = run_line.split("=", 1)[1]
    db = tmp_path / "mlflow.db"
    assert db.is_file()
    from mlflow.tracking import MlflowClient

    run = MlflowClient(tracking_uri=f"sqlite:///{db}").get_run(run_id)
    assert run.data.tags["data"] == "SYNTHETIC"
    receipt_path = Path(run.data.params["receipt_path"])
    payload = json.loads(receipt_path.read_text())
    assert payload["data_label"] == "SYNTHETIC"
