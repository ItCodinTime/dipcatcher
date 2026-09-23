"""Candle structure_finite_rate ≈ nanmean(finite_rate_* companions)."""

from __future__ import annotations

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.research.catalog import (
    candle_structure_finite_rate_covers_companions_honesty_errors,
)


def test_mismatch_fail_closed() -> None:
    errs = candle_structure_finite_rate_covers_companions_honesty_errors(
        {
            "family": "candle_order_book",
            "structure_finite_rate": 0.5,
            "finite_rate_microprice_minus_mid": 1.0,
            "finite_rate_bid_size_concentration_top": 1.0,
            "finite_rate_ask_size_concentration_top": 1.0,
        }
    )
    assert "candle_structure_finite_rate_not_nanmean_of_finite_rate_companions" in errs


def test_partial_companions_skipped() -> None:
    assert (
        candle_structure_finite_rate_covers_companions_honesty_errors(
            {
                "family": "candle_order_book",
                "structure_finite_rate": 0.5,
                "finite_rate_microprice_minus_mid": 1.0,
            }
        )
        == []
    )


def test_synth_still_clean() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert candle_structure_finite_rate_covers_companions_honesty_errors(receipt) == []
