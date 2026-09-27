"""Soft-verify northset book-shape finite_rate companions ∈ [0, 1]."""

from __future__ import annotations

from quant_fund.research.catalog import northset_book_shape_finite_rates_honesty_errors


def test_shape_rates_ok() -> None:
    assert (
        northset_book_shape_finite_rates_honesty_errors(
            {
                "concentration_top_finite_rate": 1.0,
                "queue_priority_finite_rate": 0.0,
                "side_notional_finite_rate": 0.5,
                "tob_size_share_finite_rate": 0.75,
            }
        )
        == []
    )
    assert northset_book_shape_finite_rates_honesty_errors({}) == []


def test_shape_rates_bounds() -> None:
    assert "concentration_top_finite_rate_out_of_unit_interval" in (
        northset_book_shape_finite_rates_honesty_errors({"concentration_top_finite_rate": -0.01})
    )
    assert (
        northset_book_shape_finite_rates_honesty_errors({"side_notional_finite_rate": float("nan")})
        == []
    )


def test_shape_rates_non_finite() -> None:
    assert "tob_size_share_finite_rate_non_finite" in (
        northset_book_shape_finite_rates_honesty_errors(
            {"tob_size_share_finite_rate": float("inf")}
        )
    )
