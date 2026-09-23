"""Candle mean_*_frac ∈[0,1]; imbalance_x_body ∈[-1,1]; spread_x_range ≥0."""

from __future__ import annotations

from pathlib import Path

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.research.catalog import candle_frac_and_spread_x_honesty_errors


def test_frac_bounds_fail_closed() -> None:
    base = {"family": "candle_order_book"}
    assert "mean_candle_range_frac_out_of_unit_interval" in (
        candle_frac_and_spread_x_honesty_errors({**base, "mean_candle_range_frac": 1.2})
    )
    assert "mean_candle_body_frac_out_of_unit_interval" in (
        candle_frac_and_spread_x_honesty_errors({**base, "mean_candle_body_frac": -0.1})
    )
    assert "mean_imbalance_x_body_frac_out_of_signed_unit" in (
        candle_frac_and_spread_x_honesty_errors({**base, "mean_imbalance_x_body_frac": 1.5})
    )
    assert "mean_spread_x_range_negative" in (
        candle_frac_and_spread_x_honesty_errors({**base, "mean_spread_x_range": -1.0})
    )


def test_synth_candle_frac_spread_x_clean() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert candle_frac_and_spread_x_honesty_errors(receipt) == []


def test_verify_wires_candle_frac_spread_x() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "candle_frac_and_spread_x_honesty_errors" in src
