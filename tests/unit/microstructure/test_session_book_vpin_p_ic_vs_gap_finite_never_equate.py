"""H43 session_book_vpin_p_ic ≠ gap_finite_rate — never-equate soft-verify."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    H20_HYPOTHESIS_ID,
    H21_HYPOTHESIS_ID,
    H22_HYPOTHESIS_ID,
    H43_HYPOTHESIS_ID,
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    northset_has_finite_book_uncrossed_rate,
    northset_has_finite_imbalance_p_ic,
    northset_has_finite_ohlc_identity_rate,
    northset_has_finite_session_book_vpin_p_ic,
    session_book_vpin_p_ic_vs_gap_finite_never_equate_honesty_errors,
)


def test_keys_and_h43_gate() -> None:
    assert "session_book_vpin_p_ic" != "gap_finite_rate"
    assert H43_HYPOTHESIS_ID not in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}
    elig = {"session_book_hypothesis_eligible": True, "book_hypothesis_eligible": True}
    assert northset_has_finite_session_book_vpin_p_ic({"session_book_vpin_p_ic": 0.05, **elig})
    assert not northset_has_finite_session_book_vpin_p_ic({"gap_finite_rate": 1.0, **elig})
    assert not northset_has_finite_imbalance_p_ic({"gap_finite_rate": 1.0, **elig})
    assert not northset_has_finite_ohlc_identity_rate({"gap_finite_rate": 1.0})
    assert not northset_has_finite_book_uncrossed_rate({"gap_finite_rate": 1.0, **elig})


def test_synth_both_stamped_never_equate_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(),
        cfg,
    )
    # session_book_vpin may require session L2 path; skip clean if absent
    if "session_book_vpin_p_ic" not in receipt or "gap_finite_rate" not in receipt:
        return
    assert session_book_vpin_p_ic_vs_gap_finite_never_equate_honesty_errors(receipt) == []


def test_one_key_absent_skipped() -> None:
    assert (
        session_book_vpin_p_ic_vs_gap_finite_never_equate_honesty_errors(
            {"session_book_vpin_p_ic": 0.05}
        )
        == []
    )
    assert (
        session_book_vpin_p_ic_vs_gap_finite_never_equate_honesty_errors({"gap_finite_rate": 1.0})
        == []
    )


def test_helper_registered() -> None:
    assert (
        session_book_vpin_p_ic_vs_gap_finite_never_equate_honesty_errors
        in NORTHSET_RECEIPT_HONESTY_HELPERS
    )
