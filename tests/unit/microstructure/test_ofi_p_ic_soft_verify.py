"""Soft-verify ofi_p_ic / ofi_t_ic finite when present."""

from __future__ import annotations

from quant_fund.research.catalog import ofi_p_ic_honesty_errors


def test_ofi_p_ic_ok() -> None:
    assert ofi_p_ic_honesty_errors({"ofi_p_ic": 0.2, "ofi_t_ic": 1.1}) == []
    assert ofi_p_ic_honesty_errors({}) == []


def test_ofi_p_ic_inf() -> None:
    assert "ofi_t_ic_non_finite_fail_closed" in ofi_p_ic_honesty_errors({"ofi_t_ic": float("inf")})
