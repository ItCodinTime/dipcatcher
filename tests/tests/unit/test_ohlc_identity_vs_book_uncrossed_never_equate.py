"""H20 ohlc_identity_rate ≠ H21 book_uncrossed_rate — never-equate polish."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    H20_HYPOTHESIS_ID,
    H21_HYPOTHESIS_ID,
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    northset_has_finite_book_uncrossed_rate,
    northset_has_finite_ohlc_identity_rate,
    ohlc_identity_vs_book_uncrossed_never_equate_honesty_errors,
)


def test_keys_and_gates_distinct() -> None:
    assert "ohlc_identity_rate" != "book_uncrossed_rate"
    assert H20_HYPOTHESIS_ID != H21_HYPOTHESIS_ID
    assert northset_has_finite_ohlc_identity_rate({"ohlc_identity_rate": 1.0})
    assert not northset_has_finite_ohlc_identity_rate({"book_uncrossed_rate": 1.0})
    assert northset_has_finite_book_uncrossed_rate(
        {"book_uncrossed_rate": 1.0, "book_hypothesis_eligible": True}
    )
    assert not northset_has_finite_book_uncrossed_rate(
        {"ohlc_identity_rate": 1.0, "book_hypothesis_eligible": True}
    )


def test_synth_both_stamped_never_equate_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        cfg,
    )
    assert "ohlc_identity_rate" in receipt
    assert "book_uncrossed_rate" in receipt
    assert ohlc_identity_vs_book_uncrossed_never_equate_honesty_errors(receipt) == []


def test_one_key_absent_skipped() -> None:
    assert (
        ohlc_identity_vs_book_uncrossed_never_equate_honesty_errors({"ohlc_identity_rate": 1.0})
        == []
    )


def test_helper_registered() -> None:
    assert (
        ohlc_identity_vs_book_uncrossed_never_equate_honesty_errors
        in NORTHSET_RECEIPT_HONESTY_HELPERS
    )
