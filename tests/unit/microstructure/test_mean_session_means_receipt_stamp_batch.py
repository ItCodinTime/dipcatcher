"""Batch receipt-stamp coverage for northset session-mean companions.

Individual per-key files already exist for most keys; this batch file locks the
full Lieutenant-named set + honesty dispatcher in one place, including
mean_session_book_snaps (previously only via SESSION_RECEIPT_KEYS smoke).
"""

from __future__ import annotations

import inspect
import math

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import SESSION_RECEIPT_KEYS, bench_northset
from quant_fund.research.catalog import (
    northset_session_book_snaps_honesty_errors,
    northset_session_means_honesty_errors,
)

_BATCH_KEYS = (
    "mean_session_imbalance_mean",
    "mean_session_imbalance_std",
    "mean_session_close_mid",
    "mean_session_close_micro_bps",
    "mean_session_close_imbalance",
    "mean_session_close_bid_depth",
    "mean_session_close_ask_depth",
    "mean_session_close_spread_bps",
    "mean_session_spread_bps_mean",
    "mean_session_ofi_abs_sum",
    "mean_session_book_snaps",
)


def _cfg(*, session_l2: bool) -> AppConfig:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = session_l2
    return cfg


def test_bench_source_and_session_receipt_keys_cover_batch() -> None:
    src = inspect.getsource(bench_northset)
    for key in _BATCH_KEYS:
        assert f'"{key}"' in src, key
        assert key in SESSION_RECEIPT_KEYS, key


def test_synth_session_l2_on_stamps_batch_finite_and_honest() -> None:
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(),
        _cfg(session_l2=True),
    )
    for key in _BATCH_KEYS:
        val = float(receipt[key])
        assert math.isfinite(val), key

    # Contract slices (honesty dispatcher covers the rest)
    assert -1.0 <= float(receipt["mean_session_imbalance_mean"]) <= 1.0
    assert float(receipt["mean_session_imbalance_std"]) >= 0.0
    assert float(receipt["mean_session_close_mid"]) > 0.0
    assert -1.0 <= float(receipt["mean_session_close_imbalance"]) <= 1.0
    assert float(receipt["mean_session_close_bid_depth"]) >= 0.0
    assert float(receipt["mean_session_close_ask_depth"]) >= 0.0
    assert float(receipt["mean_session_close_spread_bps"]) >= 0.0
    assert float(receipt["mean_session_spread_bps_mean"]) >= 0.0
    assert float(receipt["mean_session_ofi_abs_sum"]) >= 0.0
    assert float(receipt["mean_session_book_snaps"]) > 0.0

    assert northset_session_means_honesty_errors(receipt) == []
    assert northset_session_book_snaps_honesty_errors(receipt) == []


def test_synth_session_l2_off_batch_keys_nan() -> None:
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(),
        _cfg(session_l2=False),
    )
    for key in _BATCH_KEYS:
        assert math.isnan(float(receipt[key])), key
    # NaN skips honesty
    assert northset_session_means_honesty_errors(receipt) == []


def test_book_snaps_honesty_rejects_non_positive() -> None:
    assert northset_session_book_snaps_honesty_errors({"mean_session_book_snaps": 0.0}) == [
        "mean_session_book_snaps_non_positive_or_non_finite"
    ]
