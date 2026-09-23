"""Receipt-stamp coverage for mean_queue_priority_proxy companions (live synth)."""

from __future__ import annotations

import inspect
import math

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import mean_queue_priority_honesty_errors

_KEYS = (
    "mean_queue_priority_proxy",
    "mean_ask_queue_priority_proxy",
)


def _synth_cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    return cfg


def test_bench_source_stamps_queue_priority_means() -> None:
    src = inspect.getsource(bench_northset)
    for key in _KEYS:
        assert f'"{key}"' in src
    assert '"mean_tob_size_share"' in src


def test_synth_receipt_stamps_unit_interval_queue_priority() -> None:
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(),
        _synth_cfg(),
    )
    for key in _KEYS:
        val = float(receipt[key])
        assert math.isfinite(val) and 0.0 <= val <= 1.0, (key, val)
    assert mean_queue_priority_honesty_errors(receipt) == []


def test_honesty_rejects_out_of_unit_queue_priority() -> None:
    assert "mean_queue_priority_proxy_out_of_unit_interval" in (
        mean_queue_priority_honesty_errors({"mean_queue_priority_proxy": 1.01})
    )
    assert "mean_ask_queue_priority_proxy_out_of_unit_interval" in (
        mean_queue_priority_honesty_errors({"mean_ask_queue_priority_proxy": -0.01})
    )
