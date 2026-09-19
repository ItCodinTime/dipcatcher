"""Depth-shape finite rate on Northset + synthetic L2 emission."""

from __future__ import annotations

import pytest

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.book_metrics import DEPTH_SHAPE_FIELDS
from quant_fund.microstructure.synthetic_lob import (
    ensure_depth_shape_columns,
    synthesize_l2_from_bars,
    synthesize_session_l2,
)
from quant_fund.northset.benches import bench_northset
from quant_fund.northset.identities import session_candles_from_daily


def _bars(n_assets: int = 4, n_days: int = 24, seed: int = 5):
    return SyntheticMarketProvider(n_assets=n_assets, n_days=n_days, seed=seed).get_bars()


def test_daily_and_session_panels_emit_depth_shape() -> None:
    bars = _bars()
    daily = synthesize_l2_from_bars(bars, depth=5, seed=5)
    ensure_depth_shape_columns(daily)
    for col in DEPTH_SHAPE_FIELDS:
        assert col in daily.columns
    assert float(daily["n_bid_levels"].min()) >= 2.0
    session = session_candles_from_daily(bars, n_candles=4, seed=5)
    sess_book = synthesize_session_l2(session, depth=5, seed=5)
    ensure_depth_shape_columns(sess_book)
    assert float(sess_book["bid_log_price_slope"].drop_nans().len()) > 0


def test_ensure_depth_shape_fail_closed_on_missing() -> None:
    bars = _bars(n_assets=2, n_days=20, seed=2)
    daily = synthesize_l2_from_bars(bars, depth=5, seed=2).drop("bid_log_price_slope")
    with pytest.raises(ValueError, match="depth-shape"):
        ensure_depth_shape_columns(daily)


def test_bench_stamps_depth_shape_finite_rate() -> None:
    bars = _bars(n_assets=6, n_days=28, seed=8)
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    cfg.northset.depth_shape_finite_floor = None
    receipt = bench_northset(bars, cfg)
    assert receipt["depth_shape_finite_rate"] >= 0.99
    assert receipt["depth_shape_finite_floor"] is None


def test_bench_fail_closed_on_depth_shape_floor() -> None:
    bars = _bars(n_assets=4, n_days=24, seed=3)
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    # Impossible floor slightly above 1.0 rejected by config — use 1.0 which passes;
    # force fail by setting floor to 1.0 after monkeypatching is hard; use 1.0 OK.
    # Instead set floor=1.0 which should pass on synthetic deep books.
    cfg.northset.depth_shape_finite_floor = 1.0
    receipt = bench_northset(bars, cfg)
    assert receipt["depth_shape_finite_rate"] >= 1.0 - 1e-12

    # Now require >1 by hacking enforce: use floor=1.0 on a top-of-book-only panel path
    # via depth=1 synthetic — slopes NaN → rate NaN → fail
    cfg2 = AppConfig()
    cfg2.data.source = "synthetic"
    cfg2.northset.require_adjusted_ohlc = False
    cfg2.northset.min_names = 3
    cfg2.northset.use_session_l2 = False
    cfg2.northset.n_book_levels = 1
    cfg2.northset.depth_shape_finite_floor = 0.5
    with pytest.raises(ValueError, match="depth_shape_finite_rate"):
        bench_northset(bars, cfg2)
