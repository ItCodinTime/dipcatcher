"""Candle-book FEATURE_COLS scores structure LOB companions on the fused frame."""

from __future__ import annotations

import inspect

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure import bench as candle_bench


def test_feature_cols_include_structure_lob_companions() -> None:
    src = inspect.getsource(candle_bench)
    for col in (
        "bid_size_concentration_top",
        "ask_size_concentration_top",
        "queue_priority_proxy",
        "tob_size_share",
        "notional_imbalance",
    ):
        assert f'"{col}"' in src


def test_bench_scores_structure_lob_companions() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    receipt = candle_bench.bench_candle_order_book(bars)
    for col in (
        "bid_size_concentration_top",
        "ask_size_concentration_top",
        "queue_priority_proxy",
        "tob_size_share",
        "notional_imbalance",
    ):
        assert (
            f"{col}_spearman_ic" in receipt
            or f"{col}_ic" in receipt
            or any(col in k for k in receipt)
        ), sorted(
            k
            for k in receipt
            if "concentration" in k or "queue" in k or "tob" in k or "notional" in k
        )
