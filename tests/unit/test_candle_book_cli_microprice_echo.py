"""candle-book CLI echoes mean_microprice_minus_mid from fuse receipt."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app


def test_candle_book_cli_echoes_mean_microprice_minus_mid() -> None:
    config = Path("configs/research.yaml")
    if not config.exists():
        pytest.skip("configs/research.yaml missing")
    result = CliRunner().invoke(app, ["candle-book", "--config", str(config), "--seed", "7"])
    assert result.exit_code == 0, result.output
    assert "mean_microprice_minus_mid=" in result.output
    assert "research_only=" in result.output
