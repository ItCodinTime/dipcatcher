"""H41 reject_fold ≠ H42 follow_fold — never-equate soft-verify."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    H41_HYPOTHESIS_ID,
    H42_HYPOTHESIS_ID,
    NORTHSET_H23_H28_SPECS,
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    sweep_reject_fold_positive_fraction_vs_sweep_follow_fold_positive_fraction_never_equate_honesty_errors,
)


def test_keys_and_h41_h42_bind() -> None:
    assert "sweep_reject_fold_positive_fraction" != "sweep_follow_fold_positive_fraction"
    assert H41_HYPOTHESIS_ID != H42_HYPOTHESIS_ID
    by_key = {s[0]: s[1] for s in NORTHSET_H23_H28_SPECS}
    assert by_key["sweep_reject_fold_positive_fraction"] == H41_HYPOTHESIS_ID
    assert by_key["sweep_follow_fold_positive_fraction"] == H42_HYPOTHESIS_ID


def test_synth_both_stamped_never_equate_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(),
        cfg,
    )
    assert "sweep_reject_fold_positive_fraction" in receipt
    assert "sweep_follow_fold_positive_fraction" in receipt
    assert (
        sweep_reject_fold_positive_fraction_vs_sweep_follow_fold_positive_fraction_never_equate_honesty_errors(
            receipt
        )
        == []
    )


def test_one_key_absent_skipped() -> None:
    assert (
        sweep_reject_fold_positive_fraction_vs_sweep_follow_fold_positive_fraction_never_equate_honesty_errors(
            {"sweep_reject_fold_positive_fraction": 0.5}
        )
        == []
    )
    assert (
        sweep_reject_fold_positive_fraction_vs_sweep_follow_fold_positive_fraction_never_equate_honesty_errors(
            {"sweep_follow_fold_positive_fraction": 0.5}
        )
        == []
    )


def test_helper_registered() -> None:
    assert (
        sweep_reject_fold_positive_fraction_vs_sweep_follow_fold_positive_fraction_never_equate_honesty_errors
        in NORTHSET_RECEIPT_HONESTY_HELPERS
    )
