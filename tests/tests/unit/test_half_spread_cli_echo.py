"""northset CLI echoes mean_half_spread[_bps]."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app


def test_northset_cli_echoes_mean_half_spread() -> None:
    config = Path("configs/research.yaml")
    if not config.exists():
        pytest.skip("configs/research.yaml missing")
    result = CliRunner().invoke(app, ["northset", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "mean_half_spread=" in result.output
    assert "mean_half_spread_bps=" in result.output
    assert "amihud_mean=" in result.output
