"""Receipt-stamp coverage for queue_imbalance_mean + ofi_lag1 companions.

Prefer Lt NEXT after mean_session_* batch GREEN. Off kyle_ofi / METRICS_*;
off Sergeant structure_finite_rate / queue_priority stamp lanes.
"""

from __future__ import annotations

import inspect
import math

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    amihud_mean_honesty_errors,
    northset_lag_corr_and_sweep_count_honesty_errors,
    northset_queue_sweep_ofi_honesty_errors,
)

_KEYS = (
    "queue_imbalance_mean",
    "ofi_lag1_corr",
    "ofi_lag1_n_securities",
    "amihud_mean",
)


def _synth_cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    return cfg


def test_bench_source_stamps_queue_ofi_lag_amihud() -> None:
    src = inspect.getsource(bench_northset)
    for key in _KEYS:
        assert f'"{key}"' in src, key


def test_synth_receipt_stamps_queue_ofi_lag_amihud_honest() -> None:
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(),
        _synth_cfg(),
    )
    qi = float(receipt["queue_imbalance_mean"])
    ofi = float(receipt["ofi_lag1_corr"])
    n = float(receipt["ofi_lag1_n_securities"])
    ami = float(receipt["amihud_mean"])
    assert math.isfinite(qi) and -1.0 <= qi <= 1.0
    assert math.isfinite(ofi) and -1.0 <= ofi <= 1.0
    assert math.isfinite(n) and n >= 0.0 and float(n).is_integer()
    assert math.isfinite(ami) and ami >= 0.0
    assert northset_queue_sweep_ofi_honesty_errors(receipt) == []
    assert northset_lag_corr_and_sweep_count_honesty_errors(receipt) == []
    assert amihud_mean_honesty_errors(receipt) == []


def test_honesty_rejects_bad_queue_ofi_amihud() -> None:
    assert "queue_imbalance_mean_out_of_unit_interval" in (
        northset_queue_sweep_ofi_honesty_errors({"queue_imbalance_mean": 1.5})
    )
    assert "ofi_lag1_corr_out_of_unit_interval" in (
        northset_lag_corr_and_sweep_count_honesty_errors({"ofi_lag1_corr": -1.01})
    )
    assert amihud_mean_honesty_errors({"amihud_mean": -1e-12}) == ["amihud_mean_negative"]
