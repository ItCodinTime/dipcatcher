"""Receipt-stamp coverage for mean_bid_depth / mean_ask_depth."""

from __future__ import annotations

import inspect
import math

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import northset_depth_notional_spread_over_mid_honesty_errors


def _synth_cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    return cfg


def test_bench_source_stamps_mean_bid_ask_depth() -> None:
    src = inspect.getsource(bench_northset)
    assert '"mean_bid_depth"' in src
    assert '"mean_ask_depth"' in src


def test_synth_receipt_stamps_finite_nonneg_depths() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars()
    receipt = bench_northset(bars, _synth_cfg())
    bid = float(receipt["mean_bid_depth"])
    ask = float(receipt["mean_ask_depth"])
    assert math.isfinite(bid) and bid >= 0.0
    assert math.isfinite(ask) and ask >= 0.0
    assert northset_depth_notional_spread_over_mid_honesty_errors(receipt) == []


def test_depth_means_honesty_rejects_negative() -> None:
    assert "mean_bid_depth_negative" in northset_depth_notional_spread_over_mid_honesty_errors(
        {"mean_bid_depth": -1.0}
    )
    assert "mean_ask_depth_negative" in northset_depth_notional_spread_over_mid_honesty_errors(
        {"mean_ask_depth": -0.01}
    )
