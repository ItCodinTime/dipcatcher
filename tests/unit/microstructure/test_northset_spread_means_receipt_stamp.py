"""Receipt-stamp coverage for northset spread means companions."""

from __future__ import annotations

import inspect
import math

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    northset_half_spread_honesty_errors,
    northset_spread_means_honesty_errors,
)


def _synth_cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    return cfg


_KEYS = (
    "mean_quoted_spread",
    "mean_effective_spread",
    "mean_half_spread",
    "mean_half_spread_bps",
)


def test_bench_source_stamps_spread_means() -> None:
    src = inspect.getsource(bench_northset)
    for key in _KEYS:
        assert f'"{key}"' in src, key


def test_synth_receipt_stamps_finite_nonneg_spread_means() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=7).get_bars()
    receipt = bench_northset(bars, _synth_cfg())
    for key in _KEYS:
        val = float(receipt[key])
        assert math.isfinite(val) and val >= 0.0, key
    assert northset_spread_means_honesty_errors(receipt) == []
    assert northset_half_spread_honesty_errors(receipt) == []


def test_spread_means_honesty_rejects_negative() -> None:
    errs = northset_spread_means_honesty_errors(
        {"mean_quoted_spread": -1.0, "mean_effective_spread": -1.0}
    )
    assert "mean_quoted_spread_negative" in errs
    assert "mean_effective_spread_negative" in errs
    # half≈½quoted identity (not a negativity gate by itself)
    assert northset_half_spread_honesty_errors(
        {"mean_half_spread": 1.0, "mean_quoted_spread": 1.0}
    ) == ["mean_half_spread_not_half_of_mean_quoted_spread"]
