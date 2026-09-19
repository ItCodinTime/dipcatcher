"""Candle tob_size_share ∈(0,1]; concentration tops ∈(0,1]; log tick spacing finite."""

from __future__ import annotations

from pathlib import Path

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.research.catalog import (
    candle_log_tick_spacing_finite_honesty_errors,
    candle_ofi_qp_slope_ic_implies_mean_honesty_errors,
    mean_tob_size_share_honesty_errors,
    size_concentration_top_honesty_errors,
)


def test_fail_closed() -> None:
    assert mean_tob_size_share_honesty_errors({"mean_tob_size_share": 0.0}) == [
        "mean_tob_size_share_out_of_open_unit_interval"
    ]
    assert "mean_bid_size_concentration_top_out_of_open_unit_interval" in (
        size_concentration_top_honesty_errors({"mean_bid_size_concentration_top": 1.5})
    )
    assert candle_log_tick_spacing_finite_honesty_errors(
        {
            "family": "candle_order_book",
            "mean_bid_mean_log_tick_spacing": float("inf"),
        }
    ) == ["mean_bid_mean_log_tick_spacing_non_finite"]


def test_synth_candle_pack() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert mean_tob_size_share_honesty_errors(receipt) == []
    assert size_concentration_top_honesty_errors(receipt) == []
    assert candle_log_tick_spacing_finite_honesty_errors(receipt) == []
    assert candle_ofi_qp_slope_ic_implies_mean_honesty_errors(receipt) == []


def test_verify_wires_candle() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "candle_log_tick_spacing_finite_honesty_errors" in src
    assert (
        src.count(
            'mean_tob_size_share_honesty_errors(\n            families.get("candle_order_book")'
        )
        >= 1
    )
    assert (
        src.count(
            'size_concentration_top_honesty_errors(\n            families.get("candle_order_book")'
        )
        >= 1
    )
