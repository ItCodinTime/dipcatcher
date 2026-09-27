"""spread_over_mid FEATURE_COLS IC→mean + session_ofi_sum_* IC soft-verify."""

from __future__ import annotations

import math
from pathlib import Path

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import FEATURE_COLS, bench_candle_order_book
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_SESSION_MEANS_HONESTY_HELPERS,
    candle_feature_cols_ic_completeness_honesty_errors,
    candle_spread_over_mid_ic_implies_mean_honesty_errors,
    northset_session_ofi_sum_ic_honesty_errors,
)


def test_feature_cols_scores_spread_over_mid() -> None:
    assert "spread_over_mid" in FEATURE_COLS
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert "ic_spread_over_mid" in receipt
    assert "ic_spread_over_mid_pearson" in receipt
    mean = float(receipt["mean_spread_over_mid"])
    assert math.isfinite(mean) and mean >= 0.0
    assert candle_spread_over_mid_ic_implies_mean_honesty_errors(receipt) == []
    assert candle_feature_cols_ic_completeness_honesty_errors(receipt) == []


def test_spread_over_mid_ic_honesty_flags_missing_mean() -> None:
    assert candle_spread_over_mid_ic_implies_mean_honesty_errors(
        {"family": "candle_order_book", "ic_spread_over_mid": 0.1}
    ) == ["mean_spread_over_mid_missing_while_spread_over_mid_ic_scored"]


def test_session_ofi_sum_ic_honesty_bounds_and_synth() -> None:
    assert "session_ofi_sum_p_ic_out_of_unit_interval" in (
        northset_session_ofi_sum_ic_honesty_errors({"session_ofi_sum_p_ic": 1.5})
    )
    assert "session_ofi_sum_n_dates_negative" in (
        northset_session_ofi_sum_ic_honesty_errors({"session_ofi_sum_n_dates": -1})
    )
    assert northset_session_ofi_sum_ic_honesty_errors in NORTHSET_SESSION_MEANS_HONESTY_HELPERS
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = True
    receipt = bench_northset(SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), cfg)
    assert northset_session_ofi_sum_ic_honesty_errors(receipt) == []
    # never-equate keys: session pack ≠ daily ofi pack
    assert "session_ofi_sum_p_ic" in receipt and "ofi_p_ic" in receipt
    assert "session_ofi_sum_p_ic" != "ofi_p_ic"


def test_verify_wires_spread_over_mid_helper() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "candle_spread_over_mid_ic_implies_mean_honesty_errors" in src
