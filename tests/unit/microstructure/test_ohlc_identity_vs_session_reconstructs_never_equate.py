"""ohlc_identity_rate ≠ session_reconstructs_daily_rate — never-equate."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    H20_HYPOTHESIS_ID,
    H23_HYPOTHESIS_ID,
    NORTHSET_H23_H28_SPECS,
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    northset_has_finite_ohlc_identity_rate,
    ohlc_identity_vs_session_reconstructs_never_equate_honesty_errors,
)


def test_keys_gates_and_h23_bind() -> None:
    assert "ohlc_identity_rate" != "session_reconstructs_daily_rate"
    assert H20_HYPOTHESIS_ID != H23_HYPOTHESIS_ID
    by_key = {s[0]: s[1] for s in NORTHSET_H23_H28_SPECS}
    assert by_key["session_reconstructs_daily_rate"] == H23_HYPOTHESIS_ID
    assert northset_has_finite_ohlc_identity_rate({"ohlc_identity_rate": 1.0})
    assert not northset_has_finite_ohlc_identity_rate({"session_reconstructs_daily_rate": 1.0})


def test_synth_both_stamped_never_equate_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        cfg,
    )
    assert "ohlc_identity_rate" in receipt
    assert "session_reconstructs_daily_rate" in receipt
    assert ohlc_identity_vs_session_reconstructs_never_equate_honesty_errors(receipt) == []


def test_one_key_absent_skipped() -> None:
    assert (
        ohlc_identity_vs_session_reconstructs_never_equate_honesty_errors(
            {"ohlc_identity_rate": 1.0}
        )
        == []
    )


def test_helper_registered() -> None:
    assert (
        ohlc_identity_vs_session_reconstructs_never_equate_honesty_errors
        in NORTHSET_RECEIPT_HONESTY_HELPERS
    )
