"""sweep_reject_placebo_p ≠ sweep_follow_placebo_p — never-equate (H37≠H38)."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    H37_HYPOTHESIS_ID,
    H38_HYPOTHESIS_ID,
    NORTHSET_H23_H28_SPECS,
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    sweep_reject_placebo_p_vs_sweep_follow_placebo_p_never_equate_honesty_errors,
)


def test_keys_and_h37_h38_bind() -> None:
    assert "sweep_reject_placebo_p" != "sweep_follow_placebo_p"
    assert H37_HYPOTHESIS_ID != H38_HYPOTHESIS_ID
    by_key = {s[0]: s[1] for s in NORTHSET_H23_H28_SPECS}
    assert by_key["sweep_reject_placebo_p"] == H37_HYPOTHESIS_ID
    assert by_key["sweep_follow_placebo_p"] == H38_HYPOTHESIS_ID


def test_synth_both_stamped_never_equate_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        cfg,
    )
    assert "sweep_reject_placebo_p" in receipt
    assert "sweep_follow_placebo_p" in receipt
    assert (
        sweep_reject_placebo_p_vs_sweep_follow_placebo_p_never_equate_honesty_errors(receipt) == []
    )


def test_one_key_absent_skipped() -> None:
    assert (
        sweep_reject_placebo_p_vs_sweep_follow_placebo_p_never_equate_honesty_errors(
            {"sweep_reject_placebo_p": 0.05}
        )
        == []
    )


def test_helper_registered() -> None:
    assert (
        sweep_reject_placebo_p_vs_sweep_follow_placebo_p_never_equate_honesty_errors
        in NORTHSET_RECEIPT_HONESTY_HELPERS
    )
