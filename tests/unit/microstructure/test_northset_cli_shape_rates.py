"""Northset CLI echoes depth/concentration rates; DRY ensure wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.synthetic_lob import (
    ensure_book_panel_shape_columns,
    synthesize_l2_from_bars,
)


def test_ensure_book_panel_shape_columns_ok_and_fail() -> None:
    bars = SyntheticMarketProvider(n_assets=2, n_days=20, seed=2).get_bars()
    daily = synthesize_l2_from_bars(bars, depth=5, seed=2)
    ensure_book_panel_shape_columns(daily)
    broken = daily.drop("bid_size_concentration_top")
    with pytest.raises(ValueError, match="side-structure"):
        ensure_book_panel_shape_columns(broken)
    broken2 = daily.drop("bid_log_price_slope")
    with pytest.raises(ValueError, match="depth-shape"):
        ensure_book_panel_shape_columns(broken2)


def test_northset_cli_echoes_shape_rates(tmp_path: Path) -> None:
    # Use repo research.yaml; synthetic path is default for local lab
    config = Path("configs/research.yaml")
    if not config.exists():
        pytest.skip("configs/research.yaml missing")
    result = CliRunner().invoke(app, ["northset", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "depth_shape_finite_rate=" in result.output
    assert "concentration_top_finite_rate=" in result.output


def test_northset_cli_floor_flags_echo(tmp_path: Path) -> None:
    config = Path("configs/research.yaml")
    if not config.exists():
        pytest.skip("configs/research.yaml missing")
    result = CliRunner().invoke(
        app,
        [
            "northset",
            "--config",
            str(config),
            "--depth-shape-floor",
            "0.5",
            "--concentration-floor",
            "0.5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "depth_shape_finite_floor=0.5" in result.output
    assert "concentration_top_finite_floor=0.5" in result.output
