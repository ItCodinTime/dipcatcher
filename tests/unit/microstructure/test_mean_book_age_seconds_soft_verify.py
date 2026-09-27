"""Soft-verify mean_book_age_seconds ≥0 for northset + candle_order_book."""

from __future__ import annotations

from quant_fund.research.catalog import mean_book_age_seconds_honesty_errors


def test_book_age_nonnegative_when_finite() -> None:
    assert mean_book_age_seconds_honesty_errors({"mean_book_age_seconds": 0.0}) == []
    assert mean_book_age_seconds_honesty_errors({"mean_book_age_seconds": 12.5}) == []
    assert mean_book_age_seconds_honesty_errors({"mean_book_age_seconds": float("nan")}) == []
    assert mean_book_age_seconds_honesty_errors({}) == []


def test_book_age_flags_negative_and_inf() -> None:
    assert mean_book_age_seconds_honesty_errors({"mean_book_age_seconds": -0.01}) == [
        "mean_book_age_seconds_negative_or_non_finite"
    ]
    assert mean_book_age_seconds_honesty_errors({"mean_book_age_seconds": float("inf")}) == [
        "mean_book_age_seconds_negative_or_non_finite"
    ]


def test_book_age_family_agnostic() -> None:
    for fam in ("northset", "candle_order_book"):
        blob = {"family": fam, "mean_book_age_seconds": 1.0}
        assert mean_book_age_seconds_honesty_errors(blob) == []
