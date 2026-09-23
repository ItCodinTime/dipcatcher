"""mean_spread_bps ≥0; candle log size/price slopes finite; depth imbalance ∈[-1,1]."""

from __future__ import annotations

from pathlib import Path

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.research.catalog import (
    candle_log_slopes_finite_honesty_errors,
    candle_spread_bps_nonneg_honesty_errors,
    mean_depth_imbalance_honesty_errors,
)


def test_spread_bps_and_slopes_fail_closed() -> None:
    assert "mean_spread_bps_negative_or_non_finite" in (
        candle_spread_bps_nonneg_honesty_errors({"mean_spread_bps": -0.1})
    )
    assert "mean_bid_log_size_slope_non_finite" in (
        candle_log_slopes_finite_honesty_errors(
            {"family": "candle_order_book", "mean_bid_log_size_slope": float("inf")}
        )
    )


def test_synth_candle_depth_spread_slopes() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert mean_depth_imbalance_honesty_errors(receipt) == []
    assert candle_spread_bps_nonneg_honesty_errors(receipt) == []
    assert candle_log_slopes_finite_honesty_errors(receipt) == []


def test_verify_wires_spread_bps_and_slopes() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "candle_spread_bps_nonneg_honesty_errors" in src
    assert "candle_log_slopes_finite_honesty_errors" in src
