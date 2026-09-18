"""Candle FEATURE_COLS best_feature / mean_abs_ic identity soft-verify residual."""

from __future__ import annotations

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure import bench_candle_order_book
from quant_fund.research.catalog import candle_feature_cols_ic_honesty_errors


def test_synth_best_feature_mean_abs_identity_clean() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=40, seed=7).get_bars(),
        book=None,
        depth=5,
        seed=7,
        label="SYNTHETIC",
        min_names=3,
    )
    assert receipt.get("best_feature_ic_key")
    assert "best_feature_ic" in receipt
    assert "mean_abs_ic" in receipt
    assert candle_feature_cols_ic_honesty_errors(receipt) == []


def test_best_feature_ic_missing_despite_key_fail_closed() -> None:
    errs = candle_feature_cols_ic_honesty_errors({"ic_ofi": 0.2, "best_feature_ic_key": "ic_ofi"})
    assert "best_feature_ic_missing_despite_best_feature_ic_key" in errs


def test_best_feature_key_missing_despite_ic_fail_closed() -> None:
    errs = candle_feature_cols_ic_honesty_errors({"ic_ofi": 0.2, "best_feature_ic": 0.2})
    assert "best_feature_ic_key_missing_despite_best_feature_ic" in errs


def test_mean_abs_ic_not_mean_abs_spearman_fail_closed() -> None:
    errs = candle_feature_cols_ic_honesty_errors(
        {"ic_ofi": 0.2, "ic_spread_bps": -0.4, "mean_abs_ic": 0.05}
    )
    assert "mean_abs_ic_not_mean_abs_spearman_ic" in errs


def test_best_feature_still_rejects_not_max_abs() -> None:
    errs = candle_feature_cols_ic_honesty_errors(
        {
            "ic_ofi": 0.1,
            "ic_spread_bps": -0.5,
            "best_feature_ic_key": "ic_ofi",
            "best_feature_ic": 0.1,
        }
    )
    assert "best_feature_ic_not_max_abs_spearman_ic" in errs
