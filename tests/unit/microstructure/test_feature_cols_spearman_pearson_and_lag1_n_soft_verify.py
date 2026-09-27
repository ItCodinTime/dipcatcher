"""Honesty: FEATURE_COLS Spearman/Pearson pack completeness + lag1 n_securities."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import FEATURE_COLS, bench_candle_order_book
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    candle_feature_cols_ic_completeness_honesty_errors,
    northset_lag_corr_and_sweep_count_honesty_errors,
)


def test_completeness_requires_pearson_and_t_pack() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert candle_feature_cols_ic_completeness_honesty_errors(receipt) == []
    for col in FEATURE_COLS:
        assert f"ic_{col}_pearson" in receipt
        assert f"ic_{col}_t" in receipt


def test_completeness_flags_missing_pearson() -> None:
    blob = {
        "family": "candle_order_book",
        "n_scored": 10,
        "ic_ofi": 0.1,
        "ic_ofi_p": 0.2,
        "ic_ofi_n_dates": 3,
        # missing pearson/t and all other FEATURE_COLS keys
    }
    errs = candle_feature_cols_ic_completeness_honesty_errors(blob)
    assert "ic_ofi_pearson_missing_from_candle_feature_cols_receipt" in errs
    assert "ic_ofi_t_missing_from_candle_feature_cols_receipt" in errs


def test_lag1_n_securities_honesty_and_synth() -> None:
    assert "ofi_lag1_n_securities_negative" in (
        northset_lag_corr_and_sweep_count_honesty_errors({"ofi_lag1_n_securities": -1})
    )
    assert "mid_lag1_n_securities_non_finite" in (
        northset_lag_corr_and_sweep_count_honesty_errors({"mid_lag1_n_securities": float("inf")})
    )
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    receipt = bench_northset(SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), cfg)
    assert northset_lag_corr_and_sweep_count_honesty_errors(receipt) == []
