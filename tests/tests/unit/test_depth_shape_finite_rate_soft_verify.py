"""Soft-verify depth_shape_finite_rate ∈ [0,1]."""

from __future__ import annotations

from quant_fund.research.catalog import depth_shape_finite_rate_honesty_errors


def test_depth_shape_rate_ok_and_nan_skip() -> None:
    assert depth_shape_finite_rate_honesty_errors({"depth_shape_finite_rate": 0.0}) == []
    assert depth_shape_finite_rate_honesty_errors({"depth_shape_finite_rate": 1.0}) == []
    assert depth_shape_finite_rate_honesty_errors({"depth_shape_finite_rate": float("nan")}) == []
    assert depth_shape_finite_rate_honesty_errors({}) == []


def test_depth_shape_rate_out_of_range_and_inf() -> None:
    assert depth_shape_finite_rate_honesty_errors({"depth_shape_finite_rate": 1.01}) == [
        "depth_shape_finite_rate_out_of_unit_interval"
    ]
    assert depth_shape_finite_rate_honesty_errors({"depth_shape_finite_rate": -0.01}) == [
        "depth_shape_finite_rate_out_of_unit_interval"
    ]
    # ±inf currently fails the unit-interval gate (same error token)
    assert depth_shape_finite_rate_honesty_errors({"depth_shape_finite_rate": float("inf")}) == [
        "depth_shape_finite_rate_out_of_unit_interval"
    ]
