"""Soft-verify gap_finite_rate ∈ [0, 1] when finite."""

from __future__ import annotations

from quant_fund.research.catalog import gap_finite_rate_honesty_errors


def test_gap_finite_rate_unit_interval() -> None:
    assert gap_finite_rate_honesty_errors({"gap_finite_rate": 0.0}) == []
    assert gap_finite_rate_honesty_errors({"gap_finite_rate": 1.0}) == []
    assert gap_finite_rate_honesty_errors({"gap_finite_rate": 0.92}) == []
    assert gap_finite_rate_honesty_errors({"gap_finite_rate": float("nan")}) == []
    assert gap_finite_rate_honesty_errors({}) == []


def test_gap_finite_rate_flags_out_of_range() -> None:
    assert gap_finite_rate_honesty_errors({"gap_finite_rate": 1.01}) == [
        "gap_finite_rate_out_of_unit_interval"
    ]
    assert gap_finite_rate_honesty_errors({"gap_finite_rate": -0.01}) == [
        "gap_finite_rate_out_of_unit_interval"
    ]


def test_gap_finite_rate_rejects_inf() -> None:
    assert gap_finite_rate_honesty_errors({"gap_finite_rate": float("inf")}) == [
        "gap_finite_rate_out_of_unit_interval"
    ]
