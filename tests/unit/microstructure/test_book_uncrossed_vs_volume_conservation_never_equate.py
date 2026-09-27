"""book_uncrossed_rate ≠ session_volume_conservation_rate — never-equate (H21 ≠ H24)."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    H21_HYPOTHESIS_ID,
    H24_HYPOTHESIS_ID,
    NORTHSET_H23_H28_SPECS,
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    book_uncrossed_vs_volume_conservation_never_equate_honesty_errors,
    northset_has_finite_book_uncrossed_rate,
)


def test_keys_gates_and_h24_bind() -> None:
    assert "book_uncrossed_rate" != "session_volume_conservation_rate"
    assert H21_HYPOTHESIS_ID != H24_HYPOTHESIS_ID
    by_key = {s[0]: s[1] for s in NORTHSET_H23_H28_SPECS}
    assert by_key["session_volume_conservation_rate"] == H24_HYPOTHESIS_ID
    assert by_key.get("book_uncrossed_rate") != H24_HYPOTHESIS_ID
    assert northset_has_finite_book_uncrossed_rate(
        {"book_uncrossed_rate": 1.0, "book_hypothesis_eligible": True}
    )
    assert not northset_has_finite_book_uncrossed_rate(
        {"session_volume_conservation_rate": 1.0, "book_hypothesis_eligible": True}
    )


def test_synth_both_stamped_never_equate_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        cfg,
    )
    assert "book_uncrossed_rate" in receipt
    assert "session_volume_conservation_rate" in receipt
    assert book_uncrossed_vs_volume_conservation_never_equate_honesty_errors(receipt) == []


def test_one_key_absent_skipped() -> None:
    assert (
        book_uncrossed_vs_volume_conservation_never_equate_honesty_errors(
            {"book_uncrossed_rate": 1.0}
        )
        == []
    )


def test_helper_registered() -> None:
    assert (
        book_uncrossed_vs_volume_conservation_never_equate_honesty_errors
        in NORTHSET_RECEIPT_HONESTY_HELPERS
    )
