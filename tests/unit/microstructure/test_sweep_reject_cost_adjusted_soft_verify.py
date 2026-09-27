"""Soft-verify sweep_reject_cost_adjusted_mean_bps finite-when-present."""

from __future__ import annotations

from quant_fund.research.catalog import (
    sweep_reject_cost_adjusted_mean_bps_honesty_errors,
)


def test_reject_costed_ok() -> None:
    assert (
        sweep_reject_cost_adjusted_mean_bps_honesty_errors(
            {"sweep_reject_cost_adjusted_mean_bps": -2.5}
        )
        == []
    )
    assert (
        sweep_reject_cost_adjusted_mean_bps_honesty_errors(
            {"sweep_reject_cost_adjusted_mean_bps": 0.8}
        )
        == []
    )
    assert sweep_reject_cost_adjusted_mean_bps_honesty_errors({}) == []


def test_reject_costed_nan_skipped() -> None:
    assert (
        sweep_reject_cost_adjusted_mean_bps_honesty_errors(
            {"sweep_reject_cost_adjusted_mean_bps": float("nan")}
        )
        == []
    )


def test_reject_costed_non_finite_and_non_numeric() -> None:
    assert "sweep_reject_cost_adjusted_mean_bps_non_finite" in (
        sweep_reject_cost_adjusted_mean_bps_honesty_errors(
            {"sweep_reject_cost_adjusted_mean_bps": float("inf")}
        )
    )
    assert "sweep_reject_cost_adjusted_mean_bps_non_numeric" in (
        sweep_reject_cost_adjusted_mean_bps_honesty_errors(
            {"sweep_reject_cost_adjusted_mean_bps": "nope"}
        )
    )
