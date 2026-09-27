"""Soft-verify log size slopes finite; QLIKE means ≥0."""

from __future__ import annotations

from quant_fund.research.catalog import (
    northset_log_size_slope_honesty_errors,
    northset_qlike_means_honesty_errors,
)


def test_log_size_slopes_allow_negative_reject_inf() -> None:
    assert (
        northset_log_size_slope_honesty_errors(
            {"mean_bid_log_size_slope": -1.2, "mean_ask_log_size_slope": 0.3}
        )
        == []
    )
    assert northset_log_size_slope_honesty_errors({"mean_ask_log_size_slope": float("-inf")}) == [
        "mean_ask_log_size_slope_non_finite"
    ]


def test_qlike_nonnegative() -> None:
    assert (
        northset_qlike_means_honesty_errors(
            {"parkinson_qlike_vs_cc": 0.0, "garman_klass_qlike_vs_cc": 1.5}
        )
        == []
    )
    assert "parkinson_qlike_vs_cc_negative" in northset_qlike_means_honesty_errors(
        {"parkinson_qlike_vs_cc": -0.01}
    )


def test_qlike_nan_skip() -> None:
    assert northset_qlike_means_honesty_errors({"yang_zhang_qlike_vs_cc": float("nan")}) == []
