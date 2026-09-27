"""Soft-verify: close_imbalance ∈[-1,1]; close_mid > 0 when finite."""

from __future__ import annotations

from quant_fund.research.catalog import (
    northset_session_close_imbalance_honesty_errors,
    northset_session_close_mid_honesty_errors,
)


def test_close_imbalance_unit_interval() -> None:
    assert (
        northset_session_close_imbalance_honesty_errors({"mean_session_close_imbalance": 0.0}) == []
    )
    assert (
        northset_session_close_imbalance_honesty_errors({"mean_session_close_imbalance": 1.0}) == []
    )
    assert (
        northset_session_close_imbalance_honesty_errors({"mean_session_close_imbalance": -1.0})
        == []
    )
    assert (
        northset_session_close_imbalance_honesty_errors(
            {"mean_session_close_imbalance": float("nan")}
        )
        == []
    )
    assert northset_session_close_imbalance_honesty_errors(
        {"mean_session_close_imbalance": 1.01}
    ) == ["mean_session_close_imbalance_out_of_unit_interval"]


def test_close_mid_positive_when_finite() -> None:
    assert northset_session_close_mid_honesty_errors({"mean_session_close_mid": 100.5}) == []
    assert northset_session_close_mid_honesty_errors({"mean_session_close_mid": float("nan")}) == []
    assert northset_session_close_mid_honesty_errors({"mean_session_close_mid": 0.0}) == [
        "mean_session_close_mid_non_positive_or_non_finite"
    ]
    assert northset_session_close_mid_honesty_errors({"mean_session_close_mid": -3.0}) == [
        "mean_session_close_mid_non_positive_or_non_finite"
    ]


def test_close_mid_rejects_inf() -> None:
    assert northset_session_close_mid_honesty_errors({"mean_session_close_mid": float("inf")}) == [
        "mean_session_close_mid_non_positive_or_non_finite"
    ]
