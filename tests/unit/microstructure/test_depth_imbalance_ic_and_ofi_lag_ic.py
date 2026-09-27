"""depth_imbalance_abs FEATURE_COLS IC→mean + ofi_lag_* IC soft-verify."""

from __future__ import annotations

import math
from pathlib import Path

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import FEATURE_COLS, bench_candle_order_book
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    candle_depth_imbalance_ic_implies_mean_honesty_errors,
    candle_feature_cols_ic_completeness_honesty_errors,
    northset_ofi_lag_ic_honesty_errors,
)


def test_feature_cols_scores_depth_imbalance_abs_and_means() -> None:
    assert "depth_imbalance_abs" in FEATURE_COLS
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert "ic_depth_imbalance_abs" in receipt
    assert "ic_imbalance_depth" in receipt
    signed = float(receipt["mean_depth_imbalance"])
    abs_m = float(receipt["mean_depth_imbalance_abs"])
    assert math.isfinite(signed) and -1.0 <= signed <= 1.0
    assert math.isfinite(abs_m) and 0.0 <= abs_m <= 1.0
    assert candle_depth_imbalance_ic_implies_mean_honesty_errors(receipt) == []
    assert candle_feature_cols_ic_completeness_honesty_errors(receipt) == []


def test_depth_ic_honesty_flags_missing_means() -> None:
    assert "mean_depth_imbalance_missing_while_imbalance_depth_ic_scored" in (
        candle_depth_imbalance_ic_implies_mean_honesty_errors(
            {"family": "candle_order_book", "ic_imbalance_depth": 0.1}
        )
    )
    assert "mean_depth_imbalance_abs_missing_while_depth_imbalance_abs_ic_scored" in (
        candle_depth_imbalance_ic_implies_mean_honesty_errors(
            {"family": "candle_order_book", "ic_depth_imbalance_abs": 0.1}
        )
    )


def test_ofi_lag_ic_honesty_bounds_and_synth() -> None:
    assert "ofi_lag_p_ic_out_of_unit_interval" in (
        northset_ofi_lag_ic_honesty_errors({"ofi_lag_p_ic": 1.2})
    )
    assert "ofi_lag_n_dates_negative" in (
        northset_ofi_lag_ic_honesty_errors({"ofi_lag_n_dates": -1})
    )
    assert northset_ofi_lag_ic_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    receipt = bench_northset(SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), cfg)
    assert northset_ofi_lag_ic_honesty_errors(receipt) == []
    assert "ofi_lag_p_ic" in receipt and "ofi_p_ic" in receipt
    assert "ofi_lag_p_ic" != "ofi_p_ic"


def test_verify_wires_depth_imbalance_helper() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "candle_depth_imbalance_ic_implies_mean_honesty_errors" in src
