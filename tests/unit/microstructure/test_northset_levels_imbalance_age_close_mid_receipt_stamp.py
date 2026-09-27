"""Northset receipt stamps level/imbalance/age/close-mid means (CLI already echoes)."""

from __future__ import annotations

import inspect
import math

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset import benches

_KEYS = (
    "mean_n_bid_levels",
    "mean_n_ask_levels",
    "mean_imbalance_top",
    "mean_book_age_seconds",
    "mean_close_mid_abs_rel",
)


def _synth_receipt():
    bars = SyntheticMarketProvider(n_assets=4, n_days=40, seed=7).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    return benches.bench_northset(bars, cfg)


def test_bench_source_stamps_levels_imbalance_age_close_mid() -> None:
    src = inspect.getsource(benches.bench_northset)
    for key in _KEYS:
        assert f'"{key}"' in src, key


def test_synth_receipt_stamps_levels_imbalance_age_close_mid_present() -> None:
    receipt = _synth_receipt()
    for key in _KEYS:
        assert key in receipt, key


def test_synth_levels_age_close_mid_bounds_when_finite() -> None:
    """Stamp honesty floors only — no soft-verify re-litigation."""
    receipt = _synth_receipt()
    for key in ("mean_n_bid_levels", "mean_n_ask_levels"):
        val = float(receipt[key])
        assert math.isfinite(val), f"{key}={val}"
        assert val >= 0.0, f"{key}={val}"
    age = float(receipt["mean_book_age_seconds"])
    assert not math.isinf(age)
    if math.isfinite(age):
        assert age >= 0.0, age
    imb = float(receipt["mean_imbalance_top"])
    assert not math.isinf(imb)
    if math.isfinite(imb):
        assert -1.0 <= imb <= 1.0, imb
    cm = float(receipt["mean_close_mid_abs_rel"])
    assert not math.isinf(cm)
    if math.isfinite(cm):
        assert cm >= 0.0, cm
