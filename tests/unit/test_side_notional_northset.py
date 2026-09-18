"""Side-notional finite rate on Northset + ensure + CLI/doctor."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.book_metrics import SIDE_NOTIONAL_FIELDS
from quant_fund.microstructure.book_panel import write_book_panel
from quant_fund.microstructure.synthetic_lob import (
    ensure_book_panel_shape_columns,
    ensure_side_notional_columns,
    synthesize_l2_from_bars,
)
from quant_fund.microstructure.vendor_book_map import vendor_panel_from_bars
from quant_fund.northset.benches import bench_northset


def _bars(n_assets: int = 4, n_days: int = 24, seed: int = 5):
    return SyntheticMarketProvider(n_assets=n_assets, n_days=n_days, seed=seed).get_bars()


def test_synth_ensure_includes_side_notional() -> None:
    daily = synthesize_l2_from_bars(_bars(), depth=5, seed=5)
    ensure_side_notional_columns(daily)
    ensure_book_panel_shape_columns(daily)
    for col in SIDE_NOTIONAL_FIELDS:
        assert col in daily.columns
    broken = daily.drop("side_notional_proxy_bid")
    with pytest.raises(ValueError, match="side-notional"):
        ensure_side_notional_columns(broken)


def test_bench_stamps_side_notional_finite_rate() -> None:
    bars = _bars(n_assets=6, n_days=28, seed=8)
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    receipt = bench_northset(bars, cfg)
    assert receipt["side_notional_finite_rate"] >= 0.99
    assert receipt["side_notional_finite_floor"] is None


def test_bench_side_notional_floor_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    bars = _bars(n_assets=4, n_days=24, seed=3)
    import quant_fund.microstructure.book_metrics as bm

    monkeypatch.setattr(bm, "side_notional_finite_rate", lambda rows, **kw: 0.0)
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    cfg.northset.side_notional_finite_floor = 0.5
    with pytest.raises(ValueError, match="side_notional_finite_rate"):
        bench_northset(bars, cfg)


def test_external_missing_side_notional_nan(tmp_path: Path) -> None:
    bars = _bars(n_assets=4, n_days=28, seed=9)
    panel = vendor_panel_from_bars(bars, vendor="alpaca", seed=11)
    drop = [c for c in panel.columns if "side_notional" in c or "queue_priority" in c]
    thin = panel.drop(drop) if drop else panel
    assert "side_notional_proxy_bid" not in thin.columns
    path = write_book_panel(thin, tmp_path / "no_notional.parquet")
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    cfg.northset.book_join_coverage_floor = 0.0
    cfg.northset.book_panel_path = str(path)
    receipt = bench_northset(bars, cfg)
    assert math.isnan(float(receipt["side_notional_finite_rate"]))
    cfg.northset.side_notional_finite_floor = 0.5
    with pytest.raises(ValueError, match="side_notional_finite_rate"):
        bench_northset(bars, cfg)


def test_doctor_lists_side_notional_floor() -> None:
    config = Path("configs/research.yaml")
    if not config.exists():
        pytest.skip("configs/research.yaml missing")
    result = CliRunner().invoke(app, ["doctor", "--config", str(config)])
    assert "side_notional_finite_floor=" in result.output


def test_northset_cli_echoes_side_notional() -> None:
    config = Path("configs/research.yaml")
    if not config.exists():
        pytest.skip("configs/research.yaml missing")
    result = CliRunner().invoke(app, ["northset", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "side_notional_finite_rate=" in result.output
