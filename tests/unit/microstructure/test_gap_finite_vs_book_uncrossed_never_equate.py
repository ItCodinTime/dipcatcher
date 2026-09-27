"""gap_finite_rate ≠ H21 book_uncrossed_rate — never-equate soft-verify."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    H20_HYPOTHESIS_ID,
    H21_HYPOTHESIS_ID,
    H22_HYPOTHESIS_ID,
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    gap_finite_rate_vs_book_uncrossed_never_equate_honesty_errors,
    northset_has_finite_book_uncrossed_rate,
)


def test_keys_and_h21_gate_only_on_book() -> None:
    assert "gap_finite_rate" != "book_uncrossed_rate"
    assert H21_HYPOTHESIS_ID not in {H20_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}
    elig = {"book_hypothesis_eligible": True}
    assert northset_has_finite_book_uncrossed_rate({"book_uncrossed_rate": 1.0, **elig})
    assert not northset_has_finite_book_uncrossed_rate({"gap_finite_rate": 1.0, **elig})


def test_synth_both_stamped_never_equate_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        cfg,
    )
    assert "gap_finite_rate" in receipt
    assert "book_uncrossed_rate" in receipt
    # Numeric equality on synth is allowed; helper must stay clean
    assert gap_finite_rate_vs_book_uncrossed_never_equate_honesty_errors(receipt) == []


def test_one_key_absent_skipped() -> None:
    assert (
        gap_finite_rate_vs_book_uncrossed_never_equate_honesty_errors({"gap_finite_rate": 1.0})
        == []
    )
    assert (
        gap_finite_rate_vs_book_uncrossed_never_equate_honesty_errors({"book_uncrossed_rate": 1.0})
        == []
    )


def test_helper_registered() -> None:
    assert (
        gap_finite_rate_vs_book_uncrossed_never_equate_honesty_errors
        in NORTHSET_RECEIPT_HONESTY_HELPERS
    )
