"""tob_size_share_finite_rate on Northset + CLI/doctor."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.book_panel import write_book_panel
from quant_fund.microstructure.vendor_book_map import vendor_panel_from_bars
from quant_fund.northset.benches import bench_northset


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    return cfg


def test_bench_stamps_tob_size_share_finite_rate() -> None:
    bars = SyntheticMarketProvider(n_assets=6, n_days=28, seed=8).get_bars()
    receipt = bench_northset(bars, _cfg())
    assert receipt["tob_size_share_finite_rate"] >= 0.99
    assert receipt["tob_size_share_finite_floor"] is None
    assert math.isfinite(float(receipt["mean_tob_size_share"]))


def test_bench_tob_floor_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    import quant_fund.microstructure.book_metrics as bm

    monkeypatch.setattr(bm, "tob_size_share_finite_rate", lambda rows, **kw: 0.0)
    cfg = _cfg()
    cfg.northset.tob_size_share_finite_floor = 0.5
    with pytest.raises(ValueError, match="tob_size_share_finite_rate"):
        bench_northset(bars, cfg)


def test_external_missing_tob_share_nan(tmp_path: Path) -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=28, seed=9).get_bars()
    panel = vendor_panel_from_bars(bars, vendor="alpaca", seed=11)
    drop = [c for c in panel.columns if "tob_size_share" in c]
    thin = panel.drop(drop) if drop else panel
    assert "tob_size_share" not in thin.columns
    path = write_book_panel(thin, tmp_path / "no_tob.parquet")
    cfg = _cfg()
    cfg.northset.book_join_coverage_floor = 0.0
    cfg.northset.book_panel_path = str(path)
    receipt = bench_northset(bars, cfg)
    assert math.isnan(float(receipt["tob_size_share_finite_rate"]))
    cfg.northset.tob_size_share_finite_floor = 0.5
    with pytest.raises(ValueError, match="tob_size_share_finite_rate"):
        bench_northset(bars, cfg)


def test_doctor_and_northset_cli_echo_tob() -> None:
    config = Path("configs/research.yaml")
    if not config.exists():
        pytest.skip("configs/research.yaml missing")
    doc = CliRunner().invoke(app, ["doctor", "--config", str(config)])
    assert "tob_size_share_finite_floor=" in doc.output
    result = CliRunner().invoke(app, ["northset", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "tob_size_share_finite_rate=" in result.output
