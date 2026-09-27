"""Northset receipt stamps mean_fwd_ret_after_* follow/reclaim (CLI residual line)."""

from __future__ import annotations

import inspect
import math

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset import benches

_KEYS = (
    "mean_fwd_ret_after_high_follow",
    "mean_fwd_ret_after_high_reclaim",
    "mean_fwd_ret_after_low_follow",
    "mean_fwd_ret_after_low_reclaim",
)


def _synth_receipt():
    bars = SyntheticMarketProvider(n_assets=4, n_days=40, seed=7).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    return benches.bench_northset(bars, cfg)


def test_bench_source_stamps_fwd_ret_after_follow_reclaim() -> None:
    src = inspect.getsource(benches.bench_northset)
    for key in _KEYS:
        assert f'"{key}"' in src, key


def test_synth_receipt_stamps_fwd_ret_after_keys_present() -> None:
    receipt = _synth_receipt()
    for key in _KEYS:
        assert key in receipt, key


def test_synth_fwd_ret_after_finite_or_nan_not_inf() -> None:
    """Signed means OK; stamp floor is no ±inf (soft-verify also fail-closes inf)."""
    receipt = _synth_receipt()
    for key in _KEYS:
        val = float(receipt[key])
        assert not math.isinf(val), f"{key}={val}"
