"""Queue-priority finite rate on Northset + QUEUE_STRUCTURE ensure."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.book_metrics import QUEUE_STRUCTURE_FIELDS
from quant_fund.microstructure.book_panel import write_book_panel
from quant_fund.microstructure.synthetic_lob import (
    ensure_book_panel_shape_columns,
    ensure_queue_structure_columns,
    synthesize_l2_from_bars,
)
from quant_fund.microstructure.vendor_book_map import vendor_panel_from_bars
from quant_fund.northset.benches import bench_northset


def _bars(n_assets: int = 4, n_days: int = 24, seed: int = 5):
    return SyntheticMarketProvider(n_assets=n_assets, n_days=n_days, seed=seed).get_bars()


def test_synth_ensure_includes_queue_structure() -> None:
    daily = synthesize_l2_from_bars(_bars(), depth=5, seed=5)
    ensure_queue_structure_columns(daily)
    ensure_book_panel_shape_columns(daily)
    for col in QUEUE_STRUCTURE_FIELDS:
        assert col in daily.columns
    broken = daily.drop("queue_priority_proxy")
    with pytest.raises(ValueError, match="queue-structure"):
        ensure_queue_structure_columns(broken)
    with pytest.raises(ValueError, match="queue-structure"):
        ensure_book_panel_shape_columns(broken)


def test_bench_stamps_queue_priority_finite_rate() -> None:
    bars = _bars(n_assets=6, n_days=28, seed=8)
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    cfg.northset.queue_priority_finite_floor = None
    receipt = bench_northset(bars, cfg)
    assert receipt["queue_priority_finite_rate"] >= 0.99
    assert receipt["queue_priority_finite_floor"] is None
    assert receipt["shape_columns_ensured"] is True


def test_bench_queue_floor_pass_and_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    bars = _bars(n_assets=4, n_days=24, seed=3)
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    cfg.northset.queue_priority_finite_floor = 1.0
    receipt = bench_northset(bars, cfg)
    assert receipt["queue_priority_finite_rate"] >= 1.0 - 1e-12

    import quant_fund.microstructure.book_metrics as bm

    monkeypatch.setattr(bm, "queue_priority_finite_rate", lambda rows, **kw: 0.0)
    cfg2 = AppConfig()
    cfg2.data.source = "synthetic"
    cfg2.northset.require_adjusted_ohlc = False
    cfg2.northset.min_names = 3
    cfg2.northset.use_session_l2 = False
    cfg2.northset.queue_priority_finite_floor = 0.5
    with pytest.raises(ValueError, match="queue_priority_finite_rate"):
        bench_northset(bars, cfg2)


def test_external_missing_queue_cols_nan_and_floor(tmp_path: Path) -> None:
    bars = _bars(n_assets=4, n_days=28, seed=9)
    panel = vendor_panel_from_bars(bars, vendor="alpaca", seed=11)
    drop = [c for c in panel.columns if "queue_priority" in c or "concentration" in c]
    # also strip DEPTH_SHAPE-ish so we only care about queue honesty path
    thin = panel.drop(drop) if drop else panel
    assert "queue_priority_proxy" not in thin.columns
    path = write_book_panel(thin, tmp_path / "no_queue.parquet")
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    cfg.northset.book_join_coverage_floor = 0.0
    cfg.northset.book_panel_path = str(path)
    cfg.northset.queue_priority_finite_floor = None
    receipt = bench_northset(bars, cfg)
    assert receipt["shape_columns_ensured"] is False
    assert math.isnan(float(receipt["queue_priority_finite_rate"]))

    cfg.northset.queue_priority_finite_floor = 0.5
    with pytest.raises(ValueError, match="queue_priority_finite_rate"):
        bench_northset(bars, cfg)
