"""Candle spearman/pearson/best_feature_ic ∈[-1,1]; mean_abs_ic ∈[0,1]."""

from __future__ import annotations

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.research.catalog import candle_feature_cols_ic_honesty_errors


def test_ic_unit_fail_closed() -> None:
    assert "ic_ofi_out_of_unit_interval" in candle_feature_cols_ic_honesty_errors(
        {"family": "candle_order_book", "ic_ofi": 1.5}
    )
    assert "ic_ofi_pearson_out_of_unit_interval" in (
        candle_feature_cols_ic_honesty_errors(
            {"family": "candle_order_book", "ic_ofi_pearson": -1.2}
        )
    )
    assert "mean_abs_ic_out_of_unit_interval" in candle_feature_cols_ic_honesty_errors(
        {"family": "candle_order_book", "mean_abs_ic": 1.1}
    )
    assert "best_feature_ic_out_of_unit_interval" in (
        candle_feature_cols_ic_honesty_errors(
            {
                "family": "candle_order_book",
                "best_feature_ic_key": "ic_ofi",
                "ic_ofi": 0.2,
                "best_feature_ic": 2.0,
            }
        )
    )


def test_synth_still_clean() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert candle_feature_cols_ic_honesty_errors(receipt) == []
