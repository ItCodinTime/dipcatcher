"""Receipt-stamp coverage for mean_bid/ask_size_concentration_top."""

from __future__ import annotations

import inspect
import math

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import size_concentration_top_honesty_errors

_KEYS = (
    "mean_bid_size_concentration_top",
    "mean_ask_size_concentration_top",
)


def _synth_cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    return cfg


def test_bench_source_stamps_size_concentration_tops() -> None:
    src = inspect.getsource(bench_northset)
    for key in _KEYS:
        assert f'"{key}"' in src


def test_synth_receipt_stamps_open_unit_concentration_tops() -> None:
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(),
        _synth_cfg(),
    )
    for key in _KEYS:
        val = float(receipt[key])
        assert math.isfinite(val) and 0.0 < val <= 1.0, (key, val)
    assert size_concentration_top_honesty_errors(receipt) == []


def test_honesty_rejects_out_of_open_unit_concentration() -> None:
    assert "mean_bid_size_concentration_top_out_of_open_unit_interval" in (
        size_concentration_top_honesty_errors({"mean_bid_size_concentration_top": 0.0})
    )
    assert "mean_ask_size_concentration_top_out_of_open_unit_interval" in (
        size_concentration_top_honesty_errors({"mean_ask_size_concentration_top": 1.01})
    )
