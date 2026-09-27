"""Soft-verify book_hypothesis_eligible bool receipt flags."""

from __future__ import annotations

from quant_fund.research.catalog import book_hypothesis_eligible_honesty_errors


def test_eligible_bools_ok() -> None:
    assert (
        book_hypothesis_eligible_honesty_errors(
            {
                "book_hypothesis_eligible": True,
                "session_book_hypothesis_eligible": False,
            }
        )
        == []
    )
    assert book_hypothesis_eligible_honesty_errors({}) == []


def test_eligible_int_not_bool() -> None:
    assert "book_hypothesis_eligible_not_bool" in book_hypothesis_eligible_honesty_errors(
        {"book_hypothesis_eligible": 1}
    )
    assert "book_hypothesis_eligible_not_bool" in book_hypothesis_eligible_honesty_errors(
        {"book_hypothesis_eligible": 0}
    )


def test_session_eligible_str_not_bool() -> None:
    assert "session_book_hypothesis_eligible_not_bool" in (
        book_hypothesis_eligible_honesty_errors({"session_book_hypothesis_eligible": "False"})
    )
