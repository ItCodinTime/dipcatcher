"""Soft-verify clv_p_ic / clv_t_ic finite-when-present."""

from __future__ import annotations

from quant_fund.research.catalog import clv_p_ic_honesty_errors


def test_clv_ok() -> None:
    assert clv_p_ic_honesty_errors({"clv_p_ic": 0.05, "clv_t_ic": -2.0}) == []
    assert clv_p_ic_honesty_errors({}) == []


def test_clv_nan_skipped() -> None:
    assert clv_p_ic_honesty_errors({"clv_p_ic": float("nan")}) == []


def test_clv_non_finite_and_non_numeric() -> None:
    assert "clv_p_ic_non_finite_fail_closed" in clv_p_ic_honesty_errors({"clv_p_ic": float("inf")})
    assert "clv_t_ic_non_numeric" in clv_p_ic_honesty_errors({"clv_t_ic": "nope"})
