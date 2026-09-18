"""Always-on price_basis / return_basis stamp soft-verify."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    northset_price_return_basis_honesty_errors,
)


def test_synth_price_return_basis_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        cfg,
    )
    assert receipt.get("price_basis") == "raw_fixture_opt_out"
    assert receipt.get("return_basis") == "raw_fixture_opt_out"
    assert northset_price_return_basis_honesty_errors(receipt) == []


def test_price_basis_invalid_fail_closed() -> None:
    assert "northset_price_basis_invalid" in northset_price_return_basis_honesty_errors(
        {"family": "northset", "price_basis": "not_a_basis"}
    )


def test_raw_fixture_mismatch_fail_closed() -> None:
    assert "northset_raw_fixture_price_return_basis_mismatch" in (
        northset_price_return_basis_honesty_errors(
            {
                "family": "northset",
                "price_basis": "raw_fixture_opt_out",
                "return_basis": "split_adjusted",
            }
        )
    )


def test_split_adjusted_ok_with_total_return() -> None:
    assert (
        northset_price_return_basis_honesty_errors(
            {
                "family": "northset",
                "price_basis": "split_adjusted",
                "return_basis": "total_return",
            }
        )
        == []
    )


def test_split_adjusted_bad_return_fail_closed() -> None:
    assert "northset_split_adjusted_return_basis_invalid" in (
        northset_price_return_basis_honesty_errors(
            {
                "family": "northset",
                "price_basis": "split_adjusted",
                "return_basis": "raw_fixture_opt_out",
            }
        )
    )


def test_helper_registered() -> None:
    assert northset_price_return_basis_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
