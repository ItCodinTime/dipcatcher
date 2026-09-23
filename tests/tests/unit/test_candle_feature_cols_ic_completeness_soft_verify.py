"""Soft-verify FEATURE_COLS IC key completeness on candle_order_book receipts."""

from __future__ import annotations

from pathlib import Path

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import FEATURE_COLS, bench_candle_order_book
from quant_fund.research.catalog import candle_feature_cols_ic_completeness_honesty_errors


def test_synth_candle_receipt_has_all_feature_cols_ic_keys() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert receipt.get("family") == "candle_order_book"
    assert candle_feature_cols_ic_completeness_honesty_errors(receipt) == []
    for col in FEATURE_COLS:
        assert f"ic_{col}" in receipt


def test_completeness_flags_missing_ic_key() -> None:
    blob = {
        "family": "candle_order_book",
        "n_scored": 10,
        "ic_ofi": 0.1,
        "ic_ofi_p": 0.5,
        "ic_ofi_n_dates": 3,
    }
    errs = candle_feature_cols_ic_completeness_honesty_errors(blob)
    assert any(e.endswith("_missing_from_candle_feature_cols_receipt") for e in errs)


def test_verify_wires_completeness_helper() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "candle_feature_cols_ic_completeness_honesty_errors" in src
