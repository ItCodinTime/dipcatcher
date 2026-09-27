"""session_ohlc_identity_rate ≠ H21 book_uncrossed_rate — never-equate soft-verify."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    H20_HYPOTHESIS_ID,
    H21_HYPOTHESIS_ID,
    H23_HYPOTHESIS_ID,
    NORTHSET_H23_H28_SPECS,
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    northset_has_finite_book_uncrossed_rate,
    northset_has_finite_ohlc_identity_rate,
    session_ohlc_vs_book_uncrossed_never_equate_honesty_errors,
)


def test_keys_and_hyp_gates_distinct() -> None:
    assert "session_ohlc_identity_rate" != "book_uncrossed_rate"
    by_key = {s[0]: s[1] for s in NORTHSET_H23_H28_SPECS}
    assert by_key.get("session_ohlc_identity_rate") != H23_HYPOTHESIS_ID
    assert H21_HYPOTHESIS_ID not in {H20_HYPOTHESIS_ID, H23_HYPOTHESIS_ID}
    elig = {"book_hypothesis_eligible": True}
    assert northset_has_finite_book_uncrossed_rate({"book_uncrossed_rate": 1.0, **elig})
    assert not northset_has_finite_book_uncrossed_rate({"session_ohlc_identity_rate": 1.0, **elig})
    assert not northset_has_finite_ohlc_identity_rate({"session_ohlc_identity_rate": 1.0})
    assert not northset_has_finite_ohlc_identity_rate({"book_uncrossed_rate": 1.0})


def test_synth_both_stamped_never_equate_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.use_session_l2 = True
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        cfg,
    )
    assert "session_ohlc_identity_rate" in receipt
    assert "book_uncrossed_rate" in receipt
    assert session_ohlc_vs_book_uncrossed_never_equate_honesty_errors(receipt) == []


def test_one_key_absent_skipped() -> None:
    assert (
        session_ohlc_vs_book_uncrossed_never_equate_honesty_errors(
            {"session_ohlc_identity_rate": 1.0}
        )
        == []
    )
    assert (
        session_ohlc_vs_book_uncrossed_never_equate_honesty_errors({"book_uncrossed_rate": 1.0})
        == []
    )


def test_helper_registered() -> None:
    assert (
        session_ohlc_vs_book_uncrossed_never_equate_honesty_errors
        in NORTHSET_RECEIPT_HONESTY_HELPERS
    )
