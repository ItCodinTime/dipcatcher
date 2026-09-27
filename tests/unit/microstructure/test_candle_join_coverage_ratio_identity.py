"""Candle join_coverage ≈ n_fused / n_bars when all three present."""

from __future__ import annotations

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.research.catalog import candle_join_coverage_and_chain_honesty_errors


def test_ratio_mismatch_fail_closed() -> None:
    errs = candle_join_coverage_and_chain_honesty_errors(
        {
            "family": "candle_order_book",
            "join_coverage": 0.5,
            "n_bars": 10,
            "n_fused": 10,
            "n_scored": 8,
        }
    )
    assert "join_coverage_not_n_fused_over_n_bars" in errs


def test_ratio_match_clean() -> None:
    assert (
        candle_join_coverage_and_chain_honesty_errors(
            {
                "family": "candle_order_book",
                "join_coverage": 0.8,
                "n_bars": 10,
                "n_fused": 8,
                "n_scored": 7,
            }
        )
        == []
    )


def test_partial_skip() -> None:
    # missing n_fused → skip ratio (chain/unit checks only)
    assert "join_coverage_not_n_fused_over_n_bars" not in (
        candle_join_coverage_and_chain_honesty_errors(
            {
                "family": "candle_order_book",
                "join_coverage": 0.5,
                "n_bars": 10,
            }
        )
    )


def test_synth_still_clean() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert candle_join_coverage_and_chain_honesty_errors(receipt) == []
