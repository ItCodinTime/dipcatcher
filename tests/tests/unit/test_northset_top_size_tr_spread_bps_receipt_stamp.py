"""Receipt-stamp coverage for top sizes, true range, and mean_spread_bps."""

from __future__ import annotations

import inspect
import math

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import northset_spread_means_honesty_errors


def _synth_cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    return cfg


_KEYS = (
    "mean_top_bid_size",
    "mean_top_ask_size",
    "mean_true_range",
    "mean_spread_bps",
)


def test_bench_source_stamps_top_size_tr_spread_bps() -> None:
    src = inspect.getsource(bench_northset)
    for key in _KEYS:
        assert f'"{key}"' in src, key


def test_synth_receipt_stamps_finite_nonneg_top_size_tr_spread_bps() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=11).get_bars()
    receipt = bench_northset(bars, _synth_cfg())
    for key in _KEYS:
        val = float(receipt[key])
        assert math.isfinite(val) and val >= 0.0, key
    assert northset_spread_means_honesty_errors(receipt) == []


def test_mean_spread_bps_honesty_rejects_negative() -> None:
    assert "mean_spread_bps_negative" in northset_spread_means_honesty_errors(
        {"mean_spread_bps": -1.0}
    )
