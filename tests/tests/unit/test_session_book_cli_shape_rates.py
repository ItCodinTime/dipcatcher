"""session-book CLI echoes shape rates; ensure before write."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.synthetic_lob import ensure_book_panel_shape_columns
from quant_fund.northset.benches import bench_northset


def test_session_book_cli_echoes_shape_rates(tmp_path: Path) -> None:
    config = Path("configs/research.yaml")
    if not config.exists():
        pytest.skip("configs/research.yaml missing")
    out = tmp_path / "session_l2.parquet"
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
            "--seed",
            "7",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "depth_shape_finite_rate=" in result.output
    assert "concentration_top_finite_rate=" in result.output
    assert "shape_columns_ensured=true" in result.output


def test_session_book_ensure_fail_closed_before_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If ensure sees missing cols, CLI must fail before parquet write."""
    config = Path("configs/research.yaml")
    if not config.exists():
        pytest.skip("configs/research.yaml missing")

    def boom(panel):
        raise ValueError("L2 panel missing side-structure columns: ['bid_size_concentration_top']")

    monkeypatch.setattr(
        "quant_fund.microstructure.synthetic_lob.ensure_book_panel_shape_columns",
        boom,
    )
    # Also patch the CLI's late import path used after synthesize

    out = tmp_path / "should_not_exist.parquet"
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
    assert result.exit_code != 0
    assert not out.exists()


def test_northset_receipt_shape_columns_ensured() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = True
    receipt = bench_northset(bars, cfg)
    assert receipt["shape_columns_ensured"] is True
    # ensure helper still idempotent
    from quant_fund.microstructure.synthetic_lob import synthesize_l2_from_bars

    ensure_book_panel_shape_columns(synthesize_l2_from_bars(bars, depth=5, seed=5))
