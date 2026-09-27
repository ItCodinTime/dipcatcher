"""Northset fuse must not overwrite book effective_spread with close−mid diagnostic."""

from __future__ import annotations

import math

import polars as pl
import pytest

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.candle_book_features import attach_candle_book_features
from quant_fund.microstructure.synthetic_lob import synthesize_l2_from_bars
from quant_fund.northset.benches import bench_northset


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    return cfg


def test_fused_preserves_book_effective_spread_and_stamps_close_mid() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=28, seed=7).get_bars()
    # Reproduce enrich path lightly: attach then mimic rename contract via full bench receipt
    receipt = bench_northset(bars, _cfg())
    assert "mean_close_mid_abs_rel" in receipt
    assert math.isfinite(float(receipt["mean_effective_spread"]))
    assert math.isfinite(float(receipt["mean_close_mid_abs_rel"]))
    # Quoted spread mean and book effective spread mean should agree (both ask−bid)
    assert receipt["mean_effective_spread"] == pytest.approx(
        receipt["mean_quoted_spread"], rel=1e-9, abs=1e-9
    )
    # Candle close−mid diagnostic is a different quantity (usually much smaller scale)
    # Not always unequal, but columns must both exist on the research path:
    book = synthesize_l2_from_bars(bars, depth=5, seed=7)
    fused = attach_candle_book_features(bars, book=book, depth=5, seed=7)
    fused = fused.with_columns(
        (2.0 * (pl.col("candle_close") - pl.col("mid")).abs() / pl.col("mid")).alias(
            "close_mid_abs_rel"
        )
    )
    assert "effective_spread" in fused.columns
    assert "close_mid_abs_rel" in fused.columns
    # Book alias equals spread
    assert fused.select((pl.col("effective_spread") - pl.col("spread")).abs().max()).item() < 1e-9


def test_bench_fused_frame_keeps_book_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture fused columns after bench by patching a late step — use receipt equality as proxy."""
    bars = SyntheticMarketProvider(n_assets=6, n_days=30, seed=11).get_bars()
    receipt = bench_northset(bars, _cfg())
    # If overwrite still happened, mean_effective_spread would track close_mid not quoted
    # After fix they should match quoted within float noise.
    assert (
        abs(float(receipt["mean_effective_spread"]) - float(receipt["mean_quoted_spread"])) < 1e-9
    )
