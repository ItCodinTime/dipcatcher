"""clv_p_ic ≠ gap_finite_rate — never-equate (H30↔gap)."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    H20_HYPOTHESIS_ID,
    H21_HYPOTHESIS_ID,
    H22_HYPOTHESIS_ID,
    H30_HYPOTHESIS_ID,
    NORTHSET_H23_H28_SPECS,
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    clv_p_ic_vs_gap_finite_never_equate_honesty_errors,
    northset_has_finite_imbalance_p_ic,
)


def test_keys_and_h30_bind() -> None:
    assert "clv_p_ic" != "gap_finite_rate"
    assert H30_HYPOTHESIS_ID not in {
        H20_HYPOTHESIS_ID,
        H21_HYPOTHESIS_ID,
        H22_HYPOTHESIS_ID,
    }
    by_key = {s[0]: s[1] for s in NORTHSET_H23_H28_SPECS}
    assert by_key["clv_p_ic"] == H30_HYPOTHESIS_ID
    assert by_key.get("gap_finite_rate") != H30_HYPOTHESIS_ID
    assert not northset_has_finite_imbalance_p_ic(
        {"gap_finite_rate": 1.0, "book_hypothesis_eligible": True}
    )


def test_synth_both_stamped_never_equate_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        cfg,
    )
    assert "clv_p_ic" in receipt
    assert "gap_finite_rate" in receipt
    assert clv_p_ic_vs_gap_finite_never_equate_honesty_errors(receipt) == []


def test_one_key_absent_skipped() -> None:
    assert clv_p_ic_vs_gap_finite_never_equate_honesty_errors({"clv_p_ic": 0.05}) == []


def test_helper_registered() -> None:
    assert clv_p_ic_vs_gap_finite_never_equate_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
