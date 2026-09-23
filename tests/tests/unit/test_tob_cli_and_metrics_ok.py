"""CLI tob_size_share echo; doctor floor; metrics_required_finite_ok."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.book_panel import write_book_panel
from quant_fund.microstructure.vendor_book_map import vendor_panel_from_bars
from quant_fund.northset.benches import bench_northset


def test_doctor_lists_tob_size_share_floor() -> None:
    config = Path("configs/research.yaml")
    if not config.exists():
        pytest.skip("configs/research.yaml missing")
    result = CliRunner().invoke(app, ["doctor", "--config", str(config)])
    assert "tob_size_share_finite_floor=" in result.output
    assert "northset.shape_floors:" in result.output


def test_book_panel_cli_echoes_tob_size_share_rate(tmp_path: Path) -> None:
    config = Path("configs/research.yaml")
    if not config.exists():
        pytest.skip("configs/research.yaml missing")
    out = tmp_path / "panel.parquet"
    result = CliRunner().invoke(
        app, ["book-panel", "--config", str(config), "--out", str(out), "--depth", "5"]
    )
    assert result.exit_code == 0, result.output
    assert "tob_size_share_finite_rate=" in result.output


def test_session_book_cli_echoes_tob_size_share_rate(tmp_path: Path) -> None:
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
    assert "tob_size_share_finite_rate=" in result.output


def test_synth_metrics_required_finite_ok() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    receipt = bench_northset(bars, cfg)
    assert receipt["metrics_required_finite_ok"] is True


def test_external_thin_panel_metrics_required_finite_false(tmp_path: Path) -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=28, seed=9).get_bars()
    panel = vendor_panel_from_bars(bars, vendor="alpaca", seed=11)
    # strip several required metrics keys if present
    drop = [
        c for c in ("effective_spread", "half_spread", "quoted_spread_bps") if c in panel.columns
    ]
    thin = panel.drop(drop) if drop else panel
    path = write_book_panel(thin, tmp_path / "thin.parquet")
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    cfg.northset.book_join_coverage_floor = 0.0
    cfg.northset.book_panel_path = str(path)
    receipt = bench_northset(bars, cfg)
    assert receipt["metrics_required_finite_ok"] is False
