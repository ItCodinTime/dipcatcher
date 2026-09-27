"""Daily ohlc_identity_rate ≠ session_ohlc_identity_rate — never-equate polish."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    H20_HYPOTHESIS_ID,
    H23_HYPOTHESIS_ID,
    H24_HYPOTHESIS_ID,
    H29_HYPOTHESIS_ID,
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    ohlc_identity_vs_session_ohlc_never_equate_honesty_errors,
)


def test_keys_and_h20_distinct_from_session_binds() -> None:
    assert "ohlc_identity_rate" != "session_ohlc_identity_rate"
    assert H20_HYPOTHESIS_ID not in {
        H23_HYPOTHESIS_ID,
        H24_HYPOTHESIS_ID,
        H29_HYPOTHESIS_ID,
    }


def test_synth_both_stamped_never_equate_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.use_session_l2 = True
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        cfg,
    )
    assert "ohlc_identity_rate" in receipt
    assert "session_ohlc_identity_rate" in receipt
    assert ohlc_identity_vs_session_ohlc_never_equate_honesty_errors(receipt) == []


def test_one_key_absent_skipped() -> None:
    assert (
        ohlc_identity_vs_session_ohlc_never_equate_honesty_errors({"ohlc_identity_rate": 1.0}) == []
    )


def test_helper_registered() -> None:
    assert (
        ohlc_identity_vs_session_ohlc_never_equate_honesty_errors
        in NORTHSET_RECEIPT_HONESTY_HELPERS
    )
