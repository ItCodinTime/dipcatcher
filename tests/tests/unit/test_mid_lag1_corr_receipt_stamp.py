"""Receipt-stamp coverage for mid_lag1_corr (+ ofi_lag1 companion already covered)."""

from __future__ import annotations

import inspect
import math

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import northset_lag_corr_and_sweep_count_honesty_errors


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    return cfg


def test_bench_source_stamps_mid_lag1_corr() -> None:
    src = inspect.getsource(bench_northset)
    assert '"mid_lag1_corr"' in src
    assert '"mid_lag1_n_securities"' in src


def test_synth_receipt_stamps_mid_lag1_corr_honest() -> None:
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), _cfg()
    )
    corr = float(receipt["mid_lag1_corr"])
    n = float(receipt["mid_lag1_n_securities"])
    assert math.isfinite(corr) and -1.0 <= corr <= 1.0
    assert math.isfinite(n) and n >= 0.0
    assert northset_lag_corr_and_sweep_count_honesty_errors(receipt) == []


def test_honesty_rejects_bad_mid_lag1() -> None:
    assert "mid_lag1_corr_out_of_unit_interval" in (
        northset_lag_corr_and_sweep_count_honesty_errors({"mid_lag1_corr": 1.5})
    )
