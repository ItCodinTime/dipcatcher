"""sweep_reject_cost_adjusted_mean_bps ≠ gap_finite_rate — never-equate (H39↔gap)."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    H20_HYPOTHESIS_ID,
    H21_HYPOTHESIS_ID,
    H22_HYPOTHESIS_ID,
    H39_HYPOTHESIS_ID,
    NORTHSET_H23_H28_SPECS,
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    northset_has_finite_imbalance_p_ic,
    sweep_reject_cost_adjusted_mean_bps_vs_gap_finite_never_equate_honesty_errors,
)


def test_keys_and_h39_bind() -> None:
    assert "sweep_reject_cost_adjusted_mean_bps" != "gap_finite_rate"
    assert H39_HYPOTHESIS_ID not in {
        H20_HYPOTHESIS_ID,
        H21_HYPOTHESIS_ID,
        H22_HYPOTHESIS_ID,
    }
    by_key = {s[0]: s[1] for s in NORTHSET_H23_H28_SPECS}
    assert by_key["sweep_reject_cost_adjusted_mean_bps"] == H39_HYPOTHESIS_ID
    assert by_key.get("gap_finite_rate") != H39_HYPOTHESIS_ID
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
    assert "sweep_reject_cost_adjusted_mean_bps" in receipt
    assert "gap_finite_rate" in receipt
    assert (
        sweep_reject_cost_adjusted_mean_bps_vs_gap_finite_never_equate_honesty_errors(receipt) == []
    )


def test_one_key_absent_skipped() -> None:
    assert (
        sweep_reject_cost_adjusted_mean_bps_vs_gap_finite_never_equate_honesty_errors(
            {"sweep_reject_cost_adjusted_mean_bps": 1.25}
        )
        == []
    )


def test_helper_registered() -> None:
    assert (
        sweep_reject_cost_adjusted_mean_bps_vs_gap_finite_never_equate_honesty_errors
        in NORTHSET_RECEIPT_HONESTY_HELPERS
    )
