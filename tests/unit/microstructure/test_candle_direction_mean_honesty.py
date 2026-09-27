"""FEATURE_COLS candle_direction ternary ⇒ mean_candle_direction ∈[-1,1]."""

from __future__ import annotations

from pathlib import Path

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import FEATURE_COLS, bench_candle_order_book
from quant_fund.research.catalog import (
    candle_direction_mean_honesty_errors,
    candle_feature_cols_ic_implies_mean_honesty_errors,
)


def test_feature_cols_includes_candle_direction() -> None:
    assert "candle_direction" in FEATURE_COLS


def test_mean_candle_direction_bounds() -> None:
    assert (
        candle_direction_mean_honesty_errors(
            {"family": "candle_order_book", "mean_candle_direction": -0.2}
        )
        == []
    )
    assert "mean_candle_direction_out_of_signed_unit" in (
        candle_direction_mean_honesty_errors(
            {"family": "candle_order_book", "mean_candle_direction": 1.5}
        )
    )


def test_synth_and_ic_implies_mean_signed() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert candle_direction_mean_honesty_errors(receipt) == []
    assert candle_feature_cols_ic_implies_mean_honesty_errors(receipt) == []
    bad = dict(receipt)
    bad["mean_candle_direction"] = 2.0
    assert "mean_candle_direction_out_of_unit_interval" in (
        candle_feature_cols_ic_implies_mean_honesty_errors(bad)
    )


def test_verify_wires_candle_direction_mean() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "candle_direction_mean_honesty_errors" in src
