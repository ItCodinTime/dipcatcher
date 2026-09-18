"""Candle FEATURE_COLS date-IC / HAC meta soft-verify (ic_method + n_dates)."""

from __future__ import annotations

from pathlib import Path

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure import bench_candle_order_book
from quant_fund.research.catalog import candle_order_book_ic_method_honesty_errors


def test_synth_candle_ic_method_honesty_clean() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=40, seed=7).get_bars()
    receipt = bench_candle_order_book(bars, book=None, depth=5, seed=7, label="SYNTHETIC")
    assert receipt.get("ic_method") == "date_level_spearman_hac"
    assert any(
        isinstance(k, str)
        and k.startswith("ic_")
        and not k.endswith(("_t", "_p", "_n_dates", "_pearson"))
        for k in receipt
    )
    assert candle_order_book_ic_method_honesty_errors(receipt) == []


def test_ic_method_missing_with_feature_ic_fail_closed() -> None:
    errs = candle_order_book_ic_method_honesty_errors(
        {"family": "candle_order_book", "ic_imbalance_top": 0.1}
    )
    assert "candle_order_book_ic_method_missing" in errs


def test_ic_method_invalid_fail_closed() -> None:
    errs = candle_order_book_ic_method_honesty_errors(
        {
            "family": "candle_order_book",
            "ic_method": "not_hac",
            "ic_imbalance_top": 0.1,
        }
    )
    assert "candle_order_book_ic_method_invalid" in errs


def test_n_dates_lt_1_with_finite_ic_fail_closed() -> None:
    errs = candle_order_book_ic_method_honesty_errors(
        {
            "family": "candle_order_book",
            "ic_method": "date_level_spearman_hac",
            "ic_imbalance_top": 0.2,
            "ic_imbalance_top_n_dates": 0,
        }
    )
    assert "ic_imbalance_top_n_dates_lt_1" in errs


def test_northset_family_skipped() -> None:
    assert (
        candle_order_book_ic_method_honesty_errors({"family": "northset", "ic_imbalance_top": 0.1})
        == []
    )


def test_verify_wires_candle_ic_method_honesty() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "candle_order_book_ic_method_honesty_errors" in src
    assert 'families.get("candle_order_book")' in src
