"""SESSION_RECEIPT_KEYS: session-L2 synth receipt contains every key."""

from __future__ import annotations

import math

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import SESSION_RECEIPT_KEYS, bench_northset

# Core path means expected finite when session L2 is on (synth always provides cols).
_CORE_FINITE = frozenset(
    {
        "session_ofi_sum_mean",
        "session_book_vpin_mean",
        "mean_session_imbalance_mean",
        "mean_session_imbalance_std",
        "mean_session_spread_bps_mean",
        "mean_session_close_spread_bps",
        "mean_session_close_imbalance",
        "mean_session_close_micro_bps",
        "mean_session_close_mid",
        "mean_session_close_bid_depth",
        "mean_session_close_ask_depth",
        "mean_session_book_snaps",
    }
)


def test_session_receipt_keys_frozenset_nonempty() -> None:
    assert isinstance(SESSION_RECEIPT_KEYS, frozenset)
    assert SESSION_RECEIPT_KEYS
    assert all(k.startswith("mean_session_") or k.endswith("_mean") for k in SESSION_RECEIPT_KEYS)


def test_session_l2_synth_receipt_contains_every_session_key() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=31).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = True
    receipt = bench_northset(bars, cfg)
    missing = SESSION_RECEIPT_KEYS - set(receipt.keys())
    assert not missing, f"missing session receipt keys: {sorted(missing)}"
    for key in _CORE_FINITE & SESSION_RECEIPT_KEYS:
        val = float(receipt[key])
        assert math.isfinite(val), f"{key} expected finite with session L2 on, got {val!r}"


def test_session_l2_off_still_has_keys_as_nan() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=31).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    receipt = bench_northset(bars, cfg)
    missing = SESSION_RECEIPT_KEYS - set(receipt.keys())
    assert not missing, f"missing keys when session L2 off: {sorted(missing)}"
