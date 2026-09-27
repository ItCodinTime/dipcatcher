"""session-book/book-panel echo queue+notional; doctor floors; tob share."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset


def test_doctor_lists_all_shape_floors() -> None:
    config = Path("configs/research.yaml")
    if not config.exists():
        pytest.skip("configs/research.yaml missing")
    result = CliRunner().invoke(app, ["doctor", "--config", str(config)])
    out = result.output
    assert "northset.shape_floors:" in out
    for key in (
        "depth_shape_finite_floor=",
        "concentration_top_finite_floor=",
        "queue_priority_finite_floor=",
        "side_notional_finite_floor=",
    ):
        assert key in out


def test_book_panel_cli_echoes_queue_and_notional(tmp_path: Path) -> None:
    config = Path("configs/research.yaml")
    if not config.exists():
        pytest.skip("configs/research.yaml missing")
    out = tmp_path / "panel.parquet"
    result = CliRunner().invoke(
        app,
        ["book-panel", "--config", str(config), "--out", str(out), "--depth", "5"],
    )
    assert result.exit_code == 0, result.output
    assert "queue_priority_finite_rate=" in result.output
    assert "side_notional_finite_rate=" in result.output


def test_session_book_cli_echoes_queue_and_notional(tmp_path: Path) -> None:
    config = Path("configs/research.yaml")
    if not config.exists():
        pytest.skip("configs/research.yaml missing")
    out = tmp_path / "session.parquet"
    result = CliRunner().invoke(
        app,
        [
            "session-book",
            "--config",
            str(config),
            "--out",
            str(out),
            "--n-session",
            "4",
            "--depth",
            "5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "queue_priority_finite_rate=" in result.output
    assert "side_notional_finite_rate=" in result.output


def test_northset_stamps_mean_tob_size_share() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    receipt = bench_northset(bars, cfg)
    assert "mean_tob_size_share" in receipt
    assert math.isfinite(float(receipt["mean_tob_size_share"]))
    assert 0.0 < float(receipt["mean_tob_size_share"]) <= 1.0 + 1e-9
