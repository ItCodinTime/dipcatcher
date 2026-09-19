"""Candle |best_feature_ic| ≥ mean_abs_ic when both finite."""

from __future__ import annotations

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.research.catalog import candle_feature_cols_ic_honesty_errors


def test_best_abs_lt_mean_abs_fail_closed() -> None:
    # Feature keys present but mean_abs stamped above |best|
    errs = candle_feature_cols_ic_honesty_errors(
        {
            "family": "candle_order_book",
            "best_feature_ic_key": "ic_ofi",
            "ic_ofi": 0.1,
            "best_feature_ic": 0.1,
            "mean_abs_ic": 0.5,
        }
    )
    assert "best_feature_ic_abs_lt_mean_abs_ic" in errs


def test_best_covers_mean_without_feature_keys() -> None:
    # Aggregates alone (no ic_* spearman): still require |best| ≥ mean_abs
    clean = candle_feature_cols_ic_honesty_errors(
        {
            "family": "candle_order_book",
            "best_feature_ic": 0.4,
            "mean_abs_ic": 0.2,
        }
    )
    assert "best_feature_ic_abs_lt_mean_abs_ic" not in clean

    dirty = candle_feature_cols_ic_honesty_errors(
        {
            "family": "candle_order_book",
            "best_feature_ic": 0.1,
            "mean_abs_ic": 0.3,
        }
    )
    assert "best_feature_ic_abs_lt_mean_abs_ic" in dirty


def test_synth_still_clean() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert candle_feature_cols_ic_honesty_errors(receipt) == []
