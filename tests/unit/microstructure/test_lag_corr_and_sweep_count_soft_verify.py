"""Soft-verify mid_lag1_corr, ofi_lag1_corr, n_sweep_high."""

from __future__ import annotations

from quant_fund.research.catalog import (
    northset_lag_corr_and_sweep_count_honesty_errors,
)


def test_lag_corr_and_sweep_ok() -> None:
    assert (
        northset_lag_corr_and_sweep_count_honesty_errors(
            {
                "mid_lag1_corr": -1.0,
                "ofi_lag1_corr": 1.0,
                "n_sweep_high": 0,
            }
        )
        == []
    )
    assert northset_lag_corr_and_sweep_count_honesty_errors({}) == []


def test_lag_corr_bounds() -> None:
    assert "mid_lag1_corr_out_of_unit_interval" in (
        northset_lag_corr_and_sweep_count_honesty_errors({"mid_lag1_corr": 1.01})
    )
    assert "ofi_lag1_corr_out_of_unit_interval" in (
        northset_lag_corr_and_sweep_count_honesty_errors({"ofi_lag1_corr": -1.01})
    )
    assert northset_lag_corr_and_sweep_count_honesty_errors({"mid_lag1_corr": float("nan")}) == []


def test_n_sweep_high_non_negative() -> None:
    assert "n_sweep_high_negative" in northset_lag_corr_and_sweep_count_honesty_errors(
        {"n_sweep_high": -1}
    )
    assert "n_sweep_high_non_finite" in northset_lag_corr_and_sweep_count_honesty_errors(
        {"n_sweep_high": float("inf")}
    )
