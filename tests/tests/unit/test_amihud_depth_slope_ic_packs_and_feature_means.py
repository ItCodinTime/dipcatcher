"""Amihud/depth/slope/vpin IC packs + full FEATURE_COLS IC⇒mean on candle."""

from __future__ import annotations

import math
from pathlib import Path

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import FEATURE_COLS, bench_candle_order_book
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    candle_feature_cols_ic_completeness_honesty_errors,
    candle_feature_cols_ic_implies_mean_honesty_errors,
    northset_amihud_ic_pack_honesty_errors,
    northset_bid_log_size_slope_ic_pack_honesty_errors,
    northset_candle_body_ret_ic_pack_honesty_errors,
    northset_imbalance_depth_ic_pack_honesty_errors,
    northset_session_book_vpin_ic_pack_honesty_errors,
)


def test_candle_all_feature_cols_have_means_when_scored() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    for col in FEATURE_COLS:
        mean_key = "mean_depth_imbalance" if col == "imbalance_depth" else f"mean_{col}"
        assert mean_key in receipt, mean_key
        assert math.isfinite(float(receipt[mean_key])), mean_key
    assert candle_feature_cols_ic_implies_mean_honesty_errors(receipt) == []
    assert candle_feature_cols_ic_completeness_honesty_errors(receipt) == []


def test_ic_packs_bounds_and_synth() -> None:
    assert "amihud_p_ic_out_of_unit_interval" in (
        northset_amihud_ic_pack_honesty_errors({"amihud_p_ic": 1.5})
    )
    assert "imbalance_depth_p_ic_out_of_unit_interval" in (
        northset_imbalance_depth_ic_pack_honesty_errors({"imbalance_depth_p_ic": -0.1})
    )
    assert "bid_log_size_slope_n_dates_negative" in (
        northset_bid_log_size_slope_ic_pack_honesty_errors({"bid_log_size_slope_n_dates": -2})
    )
    assert "session_book_vpin_p_ic_out_of_unit_interval" in (
        northset_session_book_vpin_ic_pack_honesty_errors({"session_book_vpin_p_ic": 2.0})
    )
    for fn in (
        northset_amihud_ic_pack_honesty_errors,
        northset_imbalance_depth_ic_pack_honesty_errors,
        northset_bid_log_size_slope_ic_pack_honesty_errors,
        northset_candle_body_ret_ic_pack_honesty_errors,
        northset_session_book_vpin_ic_pack_honesty_errors,
    ):
        assert fn in NORTHSET_RECEIPT_HONESTY_HELPERS
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = True
    receipt = bench_northset(SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), cfg)
    assert northset_amihud_ic_pack_honesty_errors(receipt) == []
    assert northset_imbalance_depth_ic_pack_honesty_errors(receipt) == []
    assert northset_bid_log_size_slope_ic_pack_honesty_errors(receipt) == []
    assert northset_candle_body_ret_ic_pack_honesty_errors(receipt) == []
    assert northset_session_book_vpin_ic_pack_honesty_errors(receipt) == []


def test_verify_wires_feature_cols_ic_implies_mean() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "candle_feature_cols_ic_implies_mean_honesty_errors" in src
