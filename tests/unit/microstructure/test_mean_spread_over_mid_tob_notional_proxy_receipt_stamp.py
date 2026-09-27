"""Receipt-stamp coverage for depth/notional/spread_over_mid wave means."""

from __future__ import annotations

import inspect
import math

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import northset_depth_notional_spread_over_mid_honesty_errors

_KEYS = (
    "mean_spread_over_mid",
    "mean_top_of_book_notional_proxy",
    "mean_side_notional_proxy_bid",
    "mean_side_notional_proxy_ask",
)


def _synth_cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    return cfg


def test_bench_source_stamps_spread_tob_and_side_notionals() -> None:
    src = inspect.getsource(bench_northset)
    for key in _KEYS:
        assert f'"{key}"' in src


def test_synth_receipt_stamps_finite_nonneg_spread_tob_side_notionals() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars()
    receipt = bench_northset(bars, _synth_cfg())
    for key in _KEYS:
        val = float(receipt[key])
        assert math.isfinite(val) and val >= 0.0, key
    assert northset_depth_notional_spread_over_mid_honesty_errors(receipt) == []


def test_honesty_rejects_negative_spread_tob_side_notionals() -> None:
    for key in _KEYS:
        errs = northset_depth_notional_spread_over_mid_honesty_errors({key: -1e-9})
        assert f"{key}_negative" in errs, (key, errs)
