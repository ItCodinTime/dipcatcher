"""Soft-verify mean bid/ask size concentration ∈ (0,1]."""

from __future__ import annotations

from quant_fund.research.catalog import size_concentration_top_honesty_errors


def test_size_concentration_ok_and_skip() -> None:
    assert (
        size_concentration_top_honesty_errors(
            {"mean_bid_size_concentration_top": 0.4, "mean_ask_size_concentration_top": 1.0}
        )
        == []
    )
    assert size_concentration_top_honesty_errors({}) == []


def test_size_concentration_bounds() -> None:
    errs = size_concentration_top_honesty_errors({"mean_bid_size_concentration_top": 0.0})
    assert "mean_bid_size_concentration_top_out_of_open_unit_interval" in errs
    errs = size_concentration_top_honesty_errors({"mean_ask_size_concentration_top": float("inf")})
    assert "mean_ask_size_concentration_top_non_finite_fail_closed" in errs
