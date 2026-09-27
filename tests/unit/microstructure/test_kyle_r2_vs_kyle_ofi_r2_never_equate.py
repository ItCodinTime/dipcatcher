"""kyle_r2 ≠ kyle_ofi_r2 — always-on OLS R² siblings + unit soft-verify."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    kyle_r2_vs_kyle_ofi_r2_never_equate_honesty_errors,
    northset_kyle_r2_unit_honesty_errors,
)


def test_keys_distinct() -> None:
    assert "kyle_r2" != "kyle_ofi_r2"


def test_synth_both_stamped_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=40, seed=7).get_bars(),
        cfg,
    )
    assert "kyle_r2" in receipt
    assert "kyle_ofi_r2" in receipt
    assert kyle_r2_vs_kyle_ofi_r2_never_equate_honesty_errors(receipt) == []
    assert northset_kyle_r2_unit_honesty_errors(receipt) == []


def test_unit_oob_fail_closed() -> None:
    assert northset_kyle_r2_unit_honesty_errors({"kyle_r2": 1.2}) == [
        "kyle_r2_out_of_unit_interval"
    ]
    assert northset_kyle_r2_unit_honesty_errors({"kyle_ofi_r2": -0.1}) == [
        "kyle_ofi_r2_out_of_unit_interval"
    ]


def test_one_key_absent_skipped() -> None:
    assert kyle_r2_vs_kyle_ofi_r2_never_equate_honesty_errors({"kyle_r2": 0.5}) == []


def test_helpers_registered() -> None:
    assert kyle_r2_vs_kyle_ofi_r2_never_equate_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
    assert northset_kyle_r2_unit_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
