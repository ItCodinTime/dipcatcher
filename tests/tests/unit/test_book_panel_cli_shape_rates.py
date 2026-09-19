"""book-panel CLI ensure + shape rates; fail-closed before write."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app


def test_book_panel_cli_echoes_shape_rates(tmp_path: Path) -> None:
    config = Path("configs/research.yaml")
    if not config.exists():
        pytest.skip("configs/research.yaml missing")
    out = tmp_path / "synthetic_l2.parquet"
    result = CliRunner().invoke(
        app,
        [
            "book-panel",
            "--config",
            str(config),
            "--out",
            str(out),
            "--depth",
            "5",
            "--seed",
            "7",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "depth_shape_finite_rate=" in result.output
    assert "concentration_top_finite_rate=" in result.output
    assert "shape_columns_ensured=true" in result.output


def test_book_panel_ensure_fail_closed_before_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = Path("configs/research.yaml")
    if not config.exists():
        pytest.skip("configs/research.yaml missing")

    def boom(panel):
        raise ValueError("L2 panel missing depth-shape columns: ['bid_log_price_slope']")

    monkeypatch.setattr(
        "quant_fund.microstructure.synthetic_lob.ensure_book_panel_shape_columns",
        boom,
    )
    out = tmp_path / "should_not_exist.parquet"
    result = CliRunner().invoke(
        app,
        [
            "book-panel",
            "--config",
            str(config),
            "--out",
            str(out),
            "--depth",
            "5",
        ],
    )
    assert result.exit_code != 0
    assert not out.exists()


def test_doctor_lists_shape_floors() -> None:
    config = Path("configs/research.yaml")
    if not config.exists():
        pytest.skip("configs/research.yaml missing")
    result = CliRunner().invoke(app, ["doctor", "--config", str(config)])
    # doctor may exit 1 on unhealthy lab; still must print floors line
    assert "northset.shape_floors:" in result.output
    assert "depth_shape_finite_floor=" in result.output
    assert "concentration_top_finite_floor=" in result.output
