from pathlib import Path

from fastapi.testclient import TestClient

from quant_fund.api.app import app
from quant_fund.pipeline.doctor import doctor


def test_doctor() -> None:
    info = doctor("configs/research.yaml")
    assert info["core_imports"] == "ok"
    assert info["mode"] == "research"
    assert info["firm"] == "Artificial Hedge"


def test_health_endpoint() -> None:
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["firm"] == "Artificial Hedge"


def test_cli_help(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from quant_fund.cli.main import app as cli

    runner = CliRunner()
    result = runner.invoke(cli, ["doctor", "--config", "configs/research.yaml"])
    assert result.exit_code == 0
    assert "core_imports" in result.stdout
