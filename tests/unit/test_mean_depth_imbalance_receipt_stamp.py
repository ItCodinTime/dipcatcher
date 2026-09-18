"""Receipt-stamp coverage for mean_depth_imbalance (signed ∈ [-1, 1])."""

from __future__ import annotations

import inspect
import math

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import mean_depth_imbalance_honesty_errors


def _synth_cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    return cfg


def test_bench_source_stamps_mean_depth_imbalance() -> None:
    src = inspect.getsource(bench_northset)
    assert '"mean_depth_imbalance"' in src
    assert "imbalance_depth" in src


def test_synth_receipt_stamps_signed_unit_depth_imbalance() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars()
    receipt = bench_northset(bars, _synth_cfg())
    val = float(receipt["mean_depth_imbalance"])
    assert math.isfinite(val) and -1.0 <= val <= 1.0
    assert mean_depth_imbalance_honesty_errors(receipt) == []


def test_honesty_rejects_out_of_signed_unit() -> None:
    assert mean_depth_imbalance_honesty_errors({"mean_depth_imbalance": -1.1}) == [
        "mean_depth_imbalance_out_of_unit_interval"
    ]
