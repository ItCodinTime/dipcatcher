"""Soft-verify: session close depths + imbalance_std ≥0 when finite."""

from __future__ import annotations

from quant_fund.research.catalog import (
    northset_session_close_depth_honesty_errors,
    northset_session_imbalance_std_honesty_errors,
)


def test_close_depths_ok_when_nonnegative_or_nan() -> None:
    assert (
        northset_session_close_depth_honesty_errors(
            {"mean_session_close_bid_depth": 1.5, "mean_session_close_ask_depth": 0.0}
        )
        == []
    )
    assert (
        northset_session_close_depth_honesty_errors(
            {
                "mean_session_close_bid_depth": float("nan"),
                "mean_session_close_ask_depth": float("nan"),
            }
        )
        == []
    )


def test_close_depths_flag_negative() -> None:
    errs = northset_session_close_depth_honesty_errors(
        {"mean_session_close_bid_depth": -0.1, "mean_session_close_ask_depth": 2.0}
    )
    assert "mean_session_close_bid_depth_negative_or_non_finite" in errs
    errs2 = northset_session_close_depth_honesty_errors(
        {"mean_session_close_bid_depth": 1.0, "mean_session_close_ask_depth": float("-inf")}
    )
    assert "mean_session_close_ask_depth_negative_or_non_finite" in errs2


def test_imbalance_std_ok_and_flags_negative() -> None:
    assert northset_session_imbalance_std_honesty_errors({"mean_session_imbalance_std": 0.25}) == []
    assert (
        northset_session_imbalance_std_honesty_errors({"mean_session_imbalance_std": float("nan")})
        == []
    )
    errs = northset_session_imbalance_std_honesty_errors({"mean_session_imbalance_std": -1e-6})
    assert errs == ["mean_session_imbalance_std_negative_or_non_finite"]
