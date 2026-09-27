"""Soft-verify sweep_follow_cost_adjusted_mean_bps finite-when-present."""

from __future__ import annotations

from quant_fund.research.catalog import (
    sweep_follow_cost_adjusted_mean_bps_honesty_errors,
)


def test_cost_adjusted_ok_negative_and_positive() -> None:
    assert (
        sweep_follow_cost_adjusted_mean_bps_honesty_errors(
            {"sweep_follow_cost_adjusted_mean_bps": -3.2}
        )
        == []
    )
    assert (
        sweep_follow_cost_adjusted_mean_bps_honesty_errors(
            {"sweep_follow_cost_adjusted_mean_bps": 1.0}
        )
        == []
    )
    assert sweep_follow_cost_adjusted_mean_bps_honesty_errors({}) == []


def test_cost_adjusted_nan_skipped() -> None:
    assert (
        sweep_follow_cost_adjusted_mean_bps_honesty_errors(
            {"sweep_follow_cost_adjusted_mean_bps": float("nan")}
        )
        == []
    )


def test_cost_adjusted_non_finite() -> None:
    assert "sweep_follow_cost_adjusted_mean_bps_non_finite" in (
        sweep_follow_cost_adjusted_mean_bps_honesty_errors(
            {"sweep_follow_cost_adjusted_mean_bps": float("inf")}
        )
    )
