"""mean_signed_vol_x_imbalance finite when stamped (FEATURE_COLS candle)."""

from __future__ import annotations

from pathlib import Path

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import FEATURE_COLS, bench_candle_order_book
from quant_fund.research.catalog import (
    candle_feature_cols_ic_implies_mean_honesty_errors,
    candle_signed_vol_x_imbalance_mean_honesty_errors,
    candle_wick_skew_and_body_ret_means_honesty_errors,
)


def test_feature_cols_has_signed_vol_x_imbalance() -> None:
    assert "signed_vol_x_imbalance" in FEATURE_COLS


def test_signed_vol_mean_finite_fail_closed() -> None:
    assert "mean_signed_vol_x_imbalance_non_finite" in (
        candle_signed_vol_x_imbalance_mean_honesty_errors(
            {
                "family": "candle_order_book",
                "mean_signed_vol_x_imbalance": float("inf"),
            }
        )
    )


def test_synth_signed_vol_finite_and_ic_implies_mean() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert "mean_signed_vol_x_imbalance" in receipt
    assert candle_signed_vol_x_imbalance_mean_honesty_errors(receipt) == []
    assert candle_wick_skew_and_body_ret_means_honesty_errors(receipt) == []
    assert candle_feature_cols_ic_implies_mean_honesty_errors(receipt) == []


def test_verify_wires_signed_vol() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "candle_signed_vol_x_imbalance_mean_honesty_errors" in src
