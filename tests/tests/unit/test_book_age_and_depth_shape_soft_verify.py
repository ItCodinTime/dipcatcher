"""Soft-verify: book_age mean/max pair + depth_shape_finite_rate ∈[0,1]."""

from __future__ import annotations

from quant_fund.research.catalog import (
    book_age_seconds_honesty_errors,
    depth_shape_finite_rate_honesty_errors,
)


def test_book_age_max_ge_mean() -> None:
    assert (
        book_age_seconds_honesty_errors({"mean_book_age_seconds": 1.0, "max_book_age_seconds": 2.0})
        == []
    )
    assert book_age_seconds_honesty_errors(
        {"mean_book_age_seconds": 3.0, "max_book_age_seconds": 2.0}
    ) == ["max_book_age_seconds_lt_mean"]


def test_depth_shape_finite_rate_unit_interval() -> None:
    assert depth_shape_finite_rate_honesty_errors({"depth_shape_finite_rate": 0.0}) == []
    assert depth_shape_finite_rate_honesty_errors({"depth_shape_finite_rate": 1.0}) == []
    assert depth_shape_finite_rate_honesty_errors({"depth_shape_finite_rate": float("nan")}) == []
    assert depth_shape_finite_rate_honesty_errors({"depth_shape_finite_rate": 1.01}) == [
        "depth_shape_finite_rate_out_of_unit_interval"
    ]


def test_depth_shape_family_agnostic_skip_absent() -> None:
    assert depth_shape_finite_rate_honesty_errors({"family": "northset"}) == []
    assert (
        depth_shape_finite_rate_honesty_errors(
            {"family": "candle_order_book", "depth_shape_finite_rate": 0.5}
        )
        == []
    )
