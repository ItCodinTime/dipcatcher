"""Soft-verify book_uncrossed_rate ∈ [0,1]."""

from __future__ import annotations

from quant_fund.research.catalog import book_uncrossed_rate_honesty_errors


def test_book_uncrossed_rate_ok() -> None:
    assert book_uncrossed_rate_honesty_errors({"book_uncrossed_rate": 1.0}) == []
    assert book_uncrossed_rate_honesty_errors({"book_uncrossed_rate": float("nan")}) == []


def test_book_uncrossed_rate_bad() -> None:
    assert book_uncrossed_rate_honesty_errors({"book_uncrossed_rate": -0.01}) == [
        "book_uncrossed_rate_out_of_unit_interval"
    ]
