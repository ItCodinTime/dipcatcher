"""Receipt-stamp coverage for mean_microprice_weight_balance + mean_depth_imbalance_abs."""

from __future__ import annotations

import inspect
import math

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    mean_depth_imbalance_abs_honesty_errors,
    mean_microprice_weight_balance_honesty_errors,
)


def _synth_cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    return cfg


def test_bench_source_stamps_mwb_and_depth_imbalance_abs() -> None:
    src = inspect.getsource(bench_northset)
    assert '"mean_microprice_weight_balance"' in src
    assert '"mean_depth_imbalance_abs"' in src


def test_synth_receipt_stamps_unit_interval_mwb_and_abs() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars()
    receipt = bench_northset(bars, _synth_cfg())
    mwb = float(receipt["mean_microprice_weight_balance"])
    dia = float(receipt["mean_depth_imbalance_abs"])
    assert math.isfinite(mwb) and 0.0 <= mwb <= 1.0
    assert math.isfinite(dia) and 0.0 <= dia <= 1.0
    assert mean_microprice_weight_balance_honesty_errors(receipt) == []
    assert mean_depth_imbalance_abs_honesty_errors(receipt) == []


def test_honesty_rejects_out_of_unit_mwb_and_abs() -> None:
    assert mean_microprice_weight_balance_honesty_errors(
        {"mean_microprice_weight_balance": 1.01}
    ) == ["mean_microprice_weight_balance_out_of_unit_interval"]
    assert "mean_depth_imbalance_abs_out_of_unit_interval" in (
        mean_depth_imbalance_abs_honesty_errors({"mean_depth_imbalance_abs": -0.01})
    )
