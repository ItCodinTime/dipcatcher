"""Candle ``half_spread_bps`` FEATURE_COLS IC alias honesty.

``half_spread_bps == 0.5 * spread_bps`` is a monotone linear alias, so its
date-level IC pack must equal the ``spread_bps`` pack exactly (rank-identical
Spearman and affine-invariant Pearson) — a **redundant dual-IC alias**, never a
separate signal. Research diagnostic only; never live Sharpe / promotion.
"""

from __future__ import annotations

import math

import pytest

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import FEATURE_COLS, bench_candle_order_book
from quant_fund.research.catalog import candle_feature_cols_ic_completeness_honesty_errors


def test_half_spread_bps_scored_in_candle_feature_cols() -> None:
    assert "half_spread_bps" in FEATURE_COLS
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    for suffix in ("", "_pearson", "_t", "_p", "_n_dates"):
        assert f"ic_half_spread_bps{suffix}" in receipt
    assert candle_feature_cols_ic_completeness_honesty_errors(receipt) == []


def test_half_spread_bps_ic_equals_spread_bps_alias() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert math.isfinite(float(receipt["ic_half_spread_bps"]))
    assert math.isfinite(float(receipt["ic_spread_bps"]))
    # ½ monotone alias → identical IC pack (redundant, not a separate signal).
    assert float(receipt["ic_half_spread_bps"]) == pytest.approx(
        float(receipt["ic_spread_bps"]), rel=1e-9, abs=1e-12
    )
    assert float(receipt["ic_half_spread_bps_pearson"]) == pytest.approx(
        float(receipt["ic_spread_bps_pearson"]), rel=1e-9, abs=1e-12
    )
    assert float(receipt["ic_half_spread_bps_n_dates"]) == float(receipt["ic_spread_bps_n_dates"])
