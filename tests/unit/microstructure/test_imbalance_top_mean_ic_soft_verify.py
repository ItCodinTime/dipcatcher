"""Soft-verify imbalance_top_mean_ic finite when present."""

from __future__ import annotations

from quant_fund.research.catalog import imbalance_top_mean_ic_honesty_errors


def test_imbalance_top_ic_ok() -> None:
    assert imbalance_top_mean_ic_honesty_errors({"imbalance_top_mean_ic": 0.08}) == []
    assert imbalance_top_mean_ic_honesty_errors({}) == []


def test_imbalance_top_ic_inf() -> None:
    assert imbalance_top_mean_ic_honesty_errors({"imbalance_top_mean_ic": float("inf")}) == [
        "imbalance_top_mean_ic_non_finite_fail_closed"
    ]
