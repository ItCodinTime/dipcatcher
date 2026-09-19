"""Explicit soft-verify: sweep_*_fold_positive_fraction ∈[0,1]; min ∈(0,1]."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    northset_sweep_fold_positive_rates_honesty_errors,
)


def test_fold_positive_unit_and_min_positive() -> None:
    assert (
        northset_sweep_fold_positive_rates_honesty_errors(
            {
                "sweep_min_fold_positive_fraction": 0.75,
                "sweep_reject_fold_positive_fraction": 0.5,
                "sweep_follow_fold_positive_fraction": float("nan"),
            }
        )
        == []
    )
    assert "sweep_reject_fold_positive_fraction_out_of_unit_interval" in (
        northset_sweep_fold_positive_rates_honesty_errors(
            {"sweep_reject_fold_positive_fraction": 1.2}
        )
    )
    assert "sweep_min_fold_positive_fraction_not_positive_unit" in (
        northset_sweep_fold_positive_rates_honesty_errors({"sweep_min_fold_positive_fraction": 0.0})
    )


def test_synth_and_tuple_wire() -> None:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    receipt = bench_northset(SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), cfg)
    assert northset_sweep_fold_positive_rates_honesty_errors(receipt) == []
    assert northset_sweep_fold_positive_rates_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
