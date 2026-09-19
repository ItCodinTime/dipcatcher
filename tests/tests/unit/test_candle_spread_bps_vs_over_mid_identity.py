"""Candle mean_spread_bps ≈ 1e4 * mean_spread_over_mid when both finite."""

from __future__ import annotations

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.research.catalog import candle_spread_alias_honesty_errors


def test_ratio_mismatch_fail_closed() -> None:
    errs = candle_spread_alias_honesty_errors(
        {
            "family": "candle_order_book",
            "mean_spread_bps": 10.0,
            "mean_spread_over_mid": 0.0005,  # 1e4 * 0.0005 = 5 ≠ 10
        }
    )
    assert "mean_spread_bps_not_1e4_times_mean_spread_over_mid" in errs


def test_ratio_match_clean() -> None:
    assert (
        candle_spread_alias_honesty_errors(
            {
                "family": "candle_order_book",
                "mean_spread_bps": 5.0,
                "mean_spread_over_mid": 0.0005,
            }
        )
        == []
    )


def test_partial_skip() -> None:
    assert "mean_spread_bps_not_1e4_times_mean_spread_over_mid" not in (
        candle_spread_alias_honesty_errors({"family": "candle_order_book", "mean_spread_bps": 5.0})
    )


def test_synth_still_clean() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert candle_spread_alias_honesty_errors(receipt) == []
