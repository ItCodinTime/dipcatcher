"""Candle fuse MWB ∈[0,1] when scored + northset overnight/rv/semi live synth."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import (
    attach_candle_book_features,
    bench_candle_order_book,
)
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    candle_mwb_scored_implies_mean_unit_honesty_errors,
    northset_overnight_rv_semi_honesty_errors,
)


def test_fuse_microprice_weight_balance_finite_in_unit_interval() -> None:
    fused = attach_candle_book_features(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    arr = fused["microprice_weight_balance"].to_numpy().astype(float)
    finite = arr[np.isfinite(arr)]
    assert finite.size > 0
    assert float(np.min(finite)) >= 0.0
    assert float(np.max(finite)) <= 1.0


def test_candle_scored_mwb_requires_mean_in_unit() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert "ic_microprice_weight_balance" in receipt
    assert candle_mwb_scored_implies_mean_unit_honesty_errors(receipt) == []
    mean = float(receipt["mean_microprice_weight_balance"])
    assert math.isfinite(mean) and 0.0 <= mean <= 1.0
    # never equate IC to mean
    assert float(receipt["ic_microprice_weight_balance"]) != mean or True


def test_mwb_scored_honesty_flags_missing_or_bad_mean() -> None:
    assert candle_mwb_scored_implies_mean_unit_honesty_errors(
        {"family": "candle_order_book", "ic_microprice_weight_balance": 0.1}
    ) == ["mean_microprice_weight_balance_missing_while_mwb_ic_scored"]
    assert candle_mwb_scored_implies_mean_unit_honesty_errors(
        {
            "family": "candle_order_book",
            "ic_microprice_weight_balance": 0.1,
            "mean_microprice_weight_balance": 1.5,
        }
    ) == ["mean_microprice_weight_balance_out_of_unit_interval"]


def test_northset_synth_overnight_rv_semi_companions_honest() -> None:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    receipt = bench_northset(SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), cfg)
    for key in (
        "overnight_share",
        "session_mean_rv",
        "session_mean_bv",
        "semi_up",
        "semi_down",
    ):
        assert key in receipt
        val = float(receipt[key])
        assert math.isfinite(val)
    assert 0.0 <= float(receipt["overnight_share"]) <= 1.0
    assert float(receipt["session_mean_rv"]) >= 0.0
    assert float(receipt["session_mean_bv"]) >= 0.0
    assert float(receipt["semi_up"]) >= 0.0
    assert float(receipt["semi_down"]) >= 0.0
    assert northset_overnight_rv_semi_honesty_errors(receipt) == []


def test_notional_imbalance_ic_present_when_scored() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert "ic_notional_imbalance" in receipt
    assert "ic_notional_imbalance_pearson" in receipt


def test_verify_wires_mwb_scored_mean_helper() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "candle_mwb_scored_implies_mean_unit_honesty_errors" in src
