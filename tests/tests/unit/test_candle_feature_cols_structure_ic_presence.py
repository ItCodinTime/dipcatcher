"""Soft-verify candle FEATURE_COLS IC keys for structure LOB companions when scored."""

from __future__ import annotations

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import _FEATURE_COLS, bench_candle_order_book
from quant_fund.research.catalog import candle_feature_cols_ic_honesty_errors

_STRUCTURE_LOB = (
    "bid_size_concentration_top",
    "ask_size_concentration_top",
    "queue_priority_proxy",
    "tob_size_share",
    "notional_imbalance",
)


def test_feature_cols_lists_structure_lob_companions() -> None:
    for col in _STRUCTURE_LOB:
        assert col in _FEATURE_COLS


def test_synth_candle_scores_structure_lob_ic_keys() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    for col in _STRUCTURE_LOB:
        assert f"ic_{col}" in receipt, col
        assert f"ic_{col}_p" in receipt, col
        assert f"ic_{col}_n_dates" in receipt, col
        assert f"ic_{col}_t" in receipt, col
    assert candle_feature_cols_ic_honesty_errors(receipt) == []
