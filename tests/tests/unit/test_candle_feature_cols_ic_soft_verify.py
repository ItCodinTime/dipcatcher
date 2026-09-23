"""Soft-verify candle_order_book FEATURE_COLS IC receipt companions."""

from __future__ import annotations

from pathlib import Path

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import _FEATURE_COLS, bench_candle_order_book
from quant_fund.research.catalog import candle_feature_cols_ic_honesty_errors


def test_synth_candle_receipt_feature_cols_ic_honest() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    for col in _FEATURE_COLS:
        assert f"ic_{col}" in receipt
        assert f"ic_{col}_p" in receipt
        assert f"ic_{col}_n_dates" in receipt
    assert candle_feature_cols_ic_honesty_errors(receipt) == []


def test_honesty_rejects_bad_p_n_dates_mean_abs_and_best() -> None:
    assert "ic_ofi_p_out_of_unit_interval" in candle_feature_cols_ic_honesty_errors(
        {"ic_ofi_p": 1.5}
    )
    assert "ic_ofi_n_dates_negative" in candle_feature_cols_ic_honesty_errors(
        {"ic_ofi_n_dates": -1}
    )
    assert "mean_abs_ic_negative" in candle_feature_cols_ic_honesty_errors({"mean_abs_ic": -0.01})
    assert "best_feature_ic_key_not_ic_spearman_key" in candle_feature_cols_ic_honesty_errors(
        {"best_feature_ic_key": "ofi_mean_ic", "best_feature_ic": 0.1}
    )
    errs = candle_feature_cols_ic_honesty_errors(
        {
            "ic_ofi": 0.1,
            "ic_spread_bps": -0.5,
            "best_feature_ic_key": "ic_ofi",
            "best_feature_ic": 0.1,
        }
    )
    assert "best_feature_ic_not_max_abs_spearman_ic" in errs


def test_verify_wires_candle_feature_cols_ic_honesty() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "candle_feature_cols_ic_honesty_errors" in src
    assert 'families.get("candle_order_book")' in src
