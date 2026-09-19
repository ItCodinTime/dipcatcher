"""Soft-verify mean/max book_age_seconds honesty."""

from __future__ import annotations

from quant_fund.research.catalog import book_age_seconds_honesty_errors


def test_book_age_ok_and_skip_nan() -> None:
    assert (
        book_age_seconds_honesty_errors({"mean_book_age_seconds": 1.5, "max_book_age_seconds": 3.0})
        == []
    )
    assert book_age_seconds_honesty_errors({"mean_book_age_seconds": float("nan")}) == []
    assert book_age_seconds_honesty_errors({}) == []


def test_book_age_negative_and_inf_fail() -> None:
    assert "mean_book_age_seconds_negative" in book_age_seconds_honesty_errors(
        {"mean_book_age_seconds": -0.1}
    )
    assert "max_book_age_seconds_non_finite_fail_closed" in book_age_seconds_honesty_errors(
        {"max_book_age_seconds": float("inf")}
    )


def test_book_age_max_lt_mean_fail() -> None:
    errs = book_age_seconds_honesty_errors(
        {"mean_book_age_seconds": 5.0, "max_book_age_seconds": 4.0}
    )
    assert "max_book_age_seconds_lt_mean" in errs
