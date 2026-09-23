"""session_ohlc_identity_rate ≠ session_chain_rate — never-equate."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    H23_HYPOTHESIS_ID,
    H29_HYPOTHESIS_ID,
    NORTHSET_H23_H28_SPECS,
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    session_ohlc_vs_session_chain_never_equate_honesty_errors,
)


def test_keys_and_h29_bind() -> None:
    assert "session_ohlc_identity_rate" != "session_chain_rate"
    by_key = {s[0]: s[1] for s in NORTHSET_H23_H28_SPECS}
    assert by_key["session_chain_rate"] == H29_HYPOTHESIS_ID
    assert by_key.get("session_ohlc_identity_rate") != H29_HYPOTHESIS_ID
    assert by_key.get("session_ohlc_identity_rate") != H23_HYPOTHESIS_ID


def test_synth_both_stamped_never_equate_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        cfg,
    )
    assert "session_ohlc_identity_rate" in receipt
    assert "session_chain_rate" in receipt
    assert session_ohlc_vs_session_chain_never_equate_honesty_errors(receipt) == []


def test_one_key_absent_skipped() -> None:
    assert (
        session_ohlc_vs_session_chain_never_equate_honesty_errors(
            {"session_ohlc_identity_rate": 1.0}
        )
        == []
    )


def test_helper_registered() -> None:
    assert (
        session_ohlc_vs_session_chain_never_equate_honesty_errors
        in NORTHSET_RECEIPT_HONESTY_HELPERS
    )
