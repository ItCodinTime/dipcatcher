"""H21 book_uncrossed_rate ≠ H22 imbalance_top_p_ic — never-equate polish."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    H21_HYPOTHESIS_ID,
    H22_HYPOTHESIS_ID,
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    book_uncrossed_vs_imbalance_p_ic_never_equate_honesty_errors,
    northset_has_finite_book_uncrossed_rate,
    northset_has_finite_imbalance_p_ic,
)


def test_keys_and_gates_distinct() -> None:
    assert "book_uncrossed_rate" != "imbalance_top_p_ic"
    assert H21_HYPOTHESIS_ID != H22_HYPOTHESIS_ID
    elig = {"book_hypothesis_eligible": True}
    assert northset_has_finite_book_uncrossed_rate({"book_uncrossed_rate": 1.0, **elig})
    assert not northset_has_finite_book_uncrossed_rate({"imbalance_top_p_ic": 0.05, **elig})
    assert northset_has_finite_imbalance_p_ic({"imbalance_top_p_ic": 0.05, **elig})
    assert not northset_has_finite_imbalance_p_ic({"book_uncrossed_rate": 1.0, **elig})


def test_synth_both_stamped_never_equate_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(),
        cfg,
    )
    assert "book_uncrossed_rate" in receipt
    assert "imbalance_top_p_ic" in receipt
    assert book_uncrossed_vs_imbalance_p_ic_never_equate_honesty_errors(receipt) == []


def test_one_key_absent_skipped() -> None:
    assert (
        book_uncrossed_vs_imbalance_p_ic_never_equate_honesty_errors({"book_uncrossed_rate": 1.0})
        == []
    )


def test_helper_registered() -> None:
    assert (
        book_uncrossed_vs_imbalance_p_ic_never_equate_honesty_errors
        in NORTHSET_RECEIPT_HONESTY_HELPERS
    )
