"""imbalance_top_p_ic ≠ session_chain_rate — never-equate (H22↔H29)."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    H22_HYPOTHESIS_ID,
    H29_HYPOTHESIS_ID,
    NORTHSET_H23_H28_SPECS,
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    imbalance_top_p_ic_vs_session_chain_never_equate_honesty_errors,
    northset_has_finite_imbalance_p_ic,
)


def test_keys_gates_and_h29_bind() -> None:
    assert "imbalance_top_p_ic" != "session_chain_rate"
    assert H22_HYPOTHESIS_ID != H29_HYPOTHESIS_ID
    by_key = {s[0]: s[1] for s in NORTHSET_H23_H28_SPECS}
    assert by_key["session_chain_rate"] == H29_HYPOTHESIS_ID
    assert northset_has_finite_imbalance_p_ic(
        {"imbalance_top_p_ic": 0.05, "book_hypothesis_eligible": True}
    )
    assert not northset_has_finite_imbalance_p_ic(
        {"session_chain_rate": 1.0, "book_hypothesis_eligible": True}
    )


def test_synth_both_stamped_never_equate_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        cfg,
    )
    assert "imbalance_top_p_ic" in receipt
    assert "session_chain_rate" in receipt
    assert imbalance_top_p_ic_vs_session_chain_never_equate_honesty_errors(receipt) == []


def test_one_key_absent_skipped() -> None:
    assert (
        imbalance_top_p_ic_vs_session_chain_never_equate_honesty_errors(
            {"imbalance_top_p_ic": 0.05}
        )
        == []
    )


def test_helper_registered() -> None:
    assert (
        imbalance_top_p_ic_vs_session_chain_never_equate_honesty_errors
        in NORTHSET_RECEIPT_HONESTY_HELPERS
    )
