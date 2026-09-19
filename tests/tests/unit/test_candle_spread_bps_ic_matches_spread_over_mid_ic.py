"""Candle ic_spread_bps ≈ ic_spread_over_mid (1e4 monotone scale)."""

from __future__ import annotations

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.research.catalog import (
    candle_spread_bps_ic_matches_spread_over_mid_ic_honesty_errors,
)


def test_spearman_mismatch_fail_closed() -> None:
    errs = candle_spread_bps_ic_matches_spread_over_mid_ic_honesty_errors(
        {
            "family": "candle_order_book",
            "ic_spread_bps": 0.2,
            "ic_spread_over_mid": -0.1,
        }
    )
    assert "ic_spread_bps_diverges_from_ic_spread_over_mid" in errs


def test_pearson_mismatch_fail_closed() -> None:
    errs = candle_spread_bps_ic_matches_spread_over_mid_ic_honesty_errors(
        {
            "family": "candle_order_book",
            "ic_spread_bps": 0.1,
            "ic_spread_over_mid": 0.1,
            "ic_spread_bps_pearson": 0.5,
            "ic_spread_over_mid_pearson": 0.1,
        }
    )
    assert "ic_spread_bps_pearson_diverges_from_ic_spread_over_mid_pearson" in errs


def test_partial_skip() -> None:
    assert (
        candle_spread_bps_ic_matches_spread_over_mid_ic_honesty_errors(
            {"family": "candle_order_book", "ic_spread_bps": 0.2}
        )
        == []
    )


def test_synth_still_clean() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert candle_spread_bps_ic_matches_spread_over_mid_ic_honesty_errors(receipt) == []
