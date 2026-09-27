"""Concentration-top finite rate on Northset + synthetic SIDE_STRUCTURE."""

from __future__ import annotations

import pytest

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.book_metrics import SIDE_STRUCTURE_FIELDS
from quant_fund.microstructure.synthetic_lob import (
    ensure_side_structure_columns,
    synthesize_l2_from_bars,
    synthesize_session_l2,
)
from quant_fund.northset.benches import bench_northset
from quant_fund.northset.identities import session_candles_from_daily


def _bars(n_assets: int = 4, n_days: int = 24, seed: int = 5):
    return SyntheticMarketProvider(n_assets=n_assets, n_days=n_days, seed=seed).get_bars()


def test_daily_and_session_emit_side_structure() -> None:
    bars = _bars()
    daily = synthesize_l2_from_bars(bars, depth=5, seed=5)
    ensure_side_structure_columns(daily)
    for col in SIDE_STRUCTURE_FIELDS:
        assert col in daily.columns
    assert float(daily["bid_depth"].min()) > 0.0
    session = session_candles_from_daily(bars, n_candles=4, seed=5)
    sess_book = synthesize_session_l2(session, depth=5, seed=5)
    ensure_side_structure_columns(sess_book)
    assert float(sess_book["bid_size_concentration_top"].drop_nans().len()) > 0


def test_ensure_side_structure_fail_closed_on_missing() -> None:
    bars = _bars(n_assets=2, n_days=20, seed=2)
    daily = synthesize_l2_from_bars(bars, depth=5, seed=2).drop("bid_size_concentration_top")
    with pytest.raises(ValueError, match="side-structure"):
        ensure_side_structure_columns(daily)


def test_bench_stamps_concentration_top_finite_rate() -> None:
    bars = _bars(n_assets=6, n_days=28, seed=8)
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    cfg.northset.concentration_top_finite_floor = None
    receipt = bench_northset(bars, cfg)
    assert receipt["concentration_top_finite_rate"] >= 0.99
    assert receipt["concentration_top_finite_floor"] is None


def test_bench_concentration_floor_pass_and_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    bars = _bars(n_assets=4, n_days=24, seed=3)
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    cfg.northset.concentration_top_finite_floor = 1.0
    receipt = bench_northset(bars, cfg)
    assert receipt["concentration_top_finite_rate"] >= 1.0 - 1e-12

    import quant_fund.microstructure.book_metrics as bm

    monkeypatch.setattr(bm, "concentration_top_finite_rate", lambda rows, **kw: 0.0)
    cfg2 = AppConfig()
    cfg2.data.source = "synthetic"
    cfg2.northset.require_adjusted_ohlc = False
    cfg2.northset.min_names = 3
    cfg2.northset.use_session_l2 = False
    cfg2.northset.concentration_top_finite_floor = 0.5
    with pytest.raises(ValueError, match="concentration_top_finite_rate"):
        bench_northset(bars, cfg2)
