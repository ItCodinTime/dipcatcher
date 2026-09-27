"""H32 vpin_p_ic ≠ gap_finite_rate — never-equate soft-verify."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    H20_HYPOTHESIS_ID,
    H21_HYPOTHESIS_ID,
    H22_HYPOTHESIS_ID,
    H32_HYPOTHESIS_ID,
    NORTHSET_H23_H28_SPECS,
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    northset_has_finite_book_uncrossed_rate,
    northset_has_finite_imbalance_p_ic,
    northset_has_finite_ohlc_identity_rate,
    vpin_p_ic_vs_gap_finite_never_equate_honesty_errors,
)


def test_keys_and_h32_bind() -> None:
    assert "vpin_p_ic" != "gap_finite_rate"
    assert H32_HYPOTHESIS_ID not in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}
    by_key = {s[0]: s[1] for s in NORTHSET_H23_H28_SPECS}
    assert by_key["vpin_p_ic"] == H32_HYPOTHESIS_ID
    assert by_key.get("gap_finite_rate") != H32_HYPOTHESIS_ID
    elig = {"book_hypothesis_eligible": True}
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
    assert "vpin_p_ic" in receipt
    assert "gap_finite_rate" in receipt
    assert vpin_p_ic_vs_gap_finite_never_equate_honesty_errors(receipt) == []


def test_one_key_absent_skipped() -> None:
    assert vpin_p_ic_vs_gap_finite_never_equate_honesty_errors({"vpin_p_ic": 0.05}) == []
    assert vpin_p_ic_vs_gap_finite_never_equate_honesty_errors({"gap_finite_rate": 1.0}) == []


def test_helper_registered() -> None:
    assert vpin_p_ic_vs_gap_finite_never_equate_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
