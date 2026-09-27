"""Northset receipt stamps identity / session rate companions (CLI already echoes)."""

from __future__ import annotations

import inspect
import math

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset import benches
from quant_fund.research.catalog import (
    book_uncrossed_rate_honesty_errors,
    ohlc_identity_rate_honesty_errors,
    session_chain_rate_honesty_errors,
    session_reconstructs_daily_rate_honesty_errors,
    session_volume_conservation_rate_honesty_errors,
)

# Stamped on bench_northset + echoed on dipcatcher northset; soft-verify helpers exist.
# This file is the missing unit *stamp* guard (off CoS mean_bid/ask_depth lane).
_IDENTITY_RATE_KEYS = (
    "ohlc_identity_rate",
    "session_ohlc_identity_rate",
    "book_uncrossed_rate",
    "session_chain_rate",
    "session_reconstructs_daily_rate",
    "session_volume_conservation_rate",
)


def test_bench_source_stamps_identity_and_session_rates() -> None:
    src = inspect.getsource(benches.bench_northset)
    for key in _IDENTITY_RATE_KEYS:
        assert f'"{key}"' in src, key


def test_synth_receipt_stamps_identity_rates_finite_unit_interval() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=40, seed=7).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = True
    receipt = benches.bench_northset(bars, cfg)
    for key in _IDENTITY_RATE_KEYS:
        assert key in receipt, key
        val = float(receipt[key])
        assert math.isfinite(val), f"{key}={val!r}"
        assert 0.0 <= val <= 1.0, f"{key}={val}"


def test_identity_rate_soft_verify_clean_on_synth_receipt() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=40, seed=7).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = True
    receipt = benches.bench_northset(bars, cfg)
    assert ohlc_identity_rate_honesty_errors(receipt) == []
    assert book_uncrossed_rate_honesty_errors(receipt) == []
    assert session_chain_rate_honesty_errors(receipt) == []
    assert session_reconstructs_daily_rate_honesty_errors(receipt) == []
    assert session_volume_conservation_rate_honesty_errors(receipt) == []
