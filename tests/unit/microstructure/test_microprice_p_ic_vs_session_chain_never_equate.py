"""microprice_p_ic ≠ session_chain_rate — never-equate (H25↔H29)."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    H25_HYPOTHESIS_ID,
    H29_HYPOTHESIS_ID,
    NORTHSET_H23_H28_SPECS,
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    _finite_scalar,
    microprice_p_ic_vs_session_chain_never_equate_honesty_errors,
)


def test_keys_gates_and_binds() -> None:
    assert "microprice_p_ic" != "session_chain_rate"
    assert H25_HYPOTHESIS_ID != H29_HYPOTHESIS_ID
    by_key = {s[0]: s[1] for s in NORTHSET_H23_H28_SPECS}
    assert by_key["microprice_p_ic"] == H25_HYPOTHESIS_ID
    assert by_key["session_chain_rate"] == H29_HYPOTHESIS_ID
    assert _finite_scalar({"microprice_p_ic": 0.05}.get("microprice_p_ic"))
    assert not _finite_scalar({"session_chain_rate": 1.0}.get("microprice_p_ic"))


def test_synth_both_stamped_never_equate_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        cfg,
    )
    assert "microprice_p_ic" in receipt
    assert "session_chain_rate" in receipt
    assert microprice_p_ic_vs_session_chain_never_equate_honesty_errors(receipt) == []


def test_one_key_absent_skipped() -> None:
    assert (
        microprice_p_ic_vs_session_chain_never_equate_honesty_errors({"microprice_p_ic": 0.05})
        == []
    )


def test_helper_registered() -> None:
    assert (
        microprice_p_ic_vs_session_chain_never_equate_honesty_errors
        in NORTHSET_RECEIPT_HONESTY_HELPERS
    )
