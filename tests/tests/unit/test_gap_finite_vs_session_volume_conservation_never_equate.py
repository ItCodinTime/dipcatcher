"""gap_finite_rate ≠ H24 session_volume_conservation_rate — never-equate soft-verify."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    H20_HYPOTHESIS_ID,
    H21_HYPOTHESIS_ID,
    H24_HYPOTHESIS_ID,
    NORTHSET_H23_H28_SPECS,
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    gap_finite_rate_vs_session_volume_conservation_never_equate_honesty_errors,
    northset_has_finite_book_uncrossed_rate,
    northset_has_finite_ohlc_identity_rate,
)


def test_keys_and_hyp_gates_distinct() -> None:
    assert "gap_finite_rate" != "session_volume_conservation_rate"
    by_key = {s[0]: s[1] for s in NORTHSET_H23_H28_SPECS}
    assert by_key["session_volume_conservation_rate"] == H24_HYPOTHESIS_ID
    assert by_key.get("gap_finite_rate") != H24_HYPOTHESIS_ID
    assert H24_HYPOTHESIS_ID not in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID}
    elig = {"book_hypothesis_eligible": True}
    assert not northset_has_finite_book_uncrossed_rate({"gap_finite_rate": 1.0, **elig})
    assert not northset_has_finite_book_uncrossed_rate(
        {"session_volume_conservation_rate": 1.0, **elig}
    )
    assert not northset_has_finite_ohlc_identity_rate({"gap_finite_rate": 1.0})
    assert not northset_has_finite_ohlc_identity_rate({"session_volume_conservation_rate": 1.0})


def test_synth_both_stamped_never_equate_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.use_session_l2 = True
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        cfg,
    )
    assert "gap_finite_rate" in receipt
    assert "session_volume_conservation_rate" in receipt
    assert gap_finite_rate_vs_session_volume_conservation_never_equate_honesty_errors(receipt) == []


def test_one_key_absent_skipped() -> None:
    assert (
        gap_finite_rate_vs_session_volume_conservation_never_equate_honesty_errors(
            {"gap_finite_rate": 1.0}
        )
        == []
    )
    assert (
        gap_finite_rate_vs_session_volume_conservation_never_equate_honesty_errors(
            {"session_volume_conservation_rate": 1.0}
        )
        == []
    )


def test_helper_registered() -> None:
    assert (
        gap_finite_rate_vs_session_volume_conservation_never_equate_honesty_errors
        in NORTHSET_RECEIPT_HONESTY_HELPERS
    )
