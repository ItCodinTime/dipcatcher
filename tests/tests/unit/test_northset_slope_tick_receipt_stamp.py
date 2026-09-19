"""Northset receipt stamps log size/price slope + tick spacing means (CLI already echoes)."""

from __future__ import annotations

import inspect
import math

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset import benches

_SLOPE_TICK_KEYS = (
    "mean_bid_log_size_slope",
    "mean_ask_log_size_slope",
    "mean_bid_log_price_slope",
    "mean_ask_log_price_slope",
    "mean_bid_mean_log_tick_spacing",
    "mean_ask_mean_log_tick_spacing",
)


def test_bench_source_stamps_slope_and_tick_means() -> None:
    src = inspect.getsource(benches.bench_northset)
    for key in _SLOPE_TICK_KEYS:
        assert f'"{key}"' in src, key


def test_synth_receipt_stamps_slope_tick_keys_present() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=40, seed=7).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    receipt = benches.bench_northset(bars, cfg)
    for key in _SLOPE_TICK_KEYS:
        assert key in receipt, key


def test_synth_slope_keys_finite_or_nan() -> None:
    """Slopes may be signed; tick spacing is log|Δp| (may be negative when |Δp|<1).

    Stamp test only — does not re-litigate soft-verify ≥0 on tick spacing.
    """
    bars = SyntheticMarketProvider(n_assets=4, n_days=40, seed=7).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    receipt = benches.bench_northset(bars, cfg)
    for key in _SLOPE_TICK_KEYS:
        val = float(receipt[key])
        assert val == val or math.isnan(val) or math.isfinite(val)
        # reject ±inf as a stamp honesty floor (soft-verify also fail-closes inf)
        assert not math.isinf(val), f"{key}={val}"
