"""Soft-verify ofi_mean_ic finite when present."""

from __future__ import annotations

from quant_fund.research.catalog import ofi_mean_ic_honesty_errors


def test_ofi_mean_ic_ok() -> None:
    assert ofi_mean_ic_honesty_errors({"ofi_mean_ic": -0.03}) == []
    assert ofi_mean_ic_honesty_errors({}) == []


def test_ofi_mean_ic_inf() -> None:
    assert ofi_mean_ic_honesty_errors({"ofi_mean_ic": float("inf")}) == [
        "ofi_mean_ic_non_finite_fail_closed"
    ]
