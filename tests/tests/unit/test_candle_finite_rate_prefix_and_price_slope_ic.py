"""Candle finite_rate_* ∈[0,1]; bid/ask log_price_slope IC⇒mean finite."""

from __future__ import annotations

from pathlib import Path

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.research.catalog import (
    candle_all_finite_rate_prefix_honesty_errors,
    candle_ofi_qp_slope_ic_implies_mean_honesty_errors,
)


def test_finite_rate_prefix_fail_closed() -> None:
    assert "finite_rate_microprice_minus_mid_out_of_unit_interval" in (
        candle_all_finite_rate_prefix_honesty_errors(
            {
                "family": "candle_order_book",
                "finite_rate_microprice_minus_mid": 1.2,
            }
        )
    )


def test_price_slope_ic_implies_mean() -> None:
    bad = {
        "family": "candle_order_book",
        "ic_bid_log_price_slope": 0.1,
    }
    assert "mean_bid_log_price_slope_missing_while_ic_bid_log_price_slope_scored" in (
        candle_ofi_qp_slope_ic_implies_mean_honesty_errors(bad)
    )


def test_synth_and_verify() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert candle_all_finite_rate_prefix_honesty_errors(receipt) == []
    assert candle_ofi_qp_slope_ic_implies_mean_honesty_errors(receipt) == []
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "candle_all_finite_rate_prefix_honesty_errors" in src
    assert "ic_bid_log_price_slope" in Path("src/quant_fund/research/catalog.py").read_text(
        encoding="utf-8"
    )
