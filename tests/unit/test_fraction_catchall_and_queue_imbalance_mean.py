"""Honesty: *_fraction unit catch-all + queue_imbalance_mean alias."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    northset_all_fraction_unit_honesty_errors,
    northset_queue_imbalance_mean_alias_honesty_errors,
)


def test_fraction_unit_ok_and_oob() -> None:
    assert (
        northset_all_fraction_unit_honesty_errors(
            {"sweep_min_fold_positive_fraction": 0.75, "other": 9}
        )
        == []
    )
    assert "sweep_min_fold_positive_fraction_out_of_unit_interval" in (
        northset_all_fraction_unit_honesty_errors({"sweep_min_fold_positive_fraction": 1.5})
    )
    assert (
        northset_all_fraction_unit_honesty_errors(
            {"sweep_reject_fold_positive_fraction": float("nan")}
        )
        == []
    )


def test_queue_imbalance_mean_alias() -> None:
    assert northset_queue_imbalance_mean_alias_honesty_errors({"queue_imbalance_mean": 0.1}) == []
    assert northset_queue_imbalance_mean_alias_honesty_errors({}) == []
    assert "queue_imbalance_mean_out_of_signed_unit" in (
        northset_queue_imbalance_mean_alias_honesty_errors({"queue_imbalance_mean": 2.0})
    )


def test_synth_receipt_fraction_and_queue_honest() -> None:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    nr = bench_northset(SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), cfg)
    assert northset_all_fraction_unit_honesty_errors(nr) == []
    assert northset_queue_imbalance_mean_alias_honesty_errors(nr) == []
    assert northset_all_fraction_unit_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
    assert northset_queue_imbalance_mean_alias_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
