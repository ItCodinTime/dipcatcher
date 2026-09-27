"""Soft-verify sweep_follow_event_mean_bps finite-when-present."""

from __future__ import annotations

from quant_fund.research.catalog import sweep_follow_event_mean_bps_honesty_errors


def test_follow_event_ok() -> None:
    assert sweep_follow_event_mean_bps_honesty_errors({"sweep_follow_event_mean_bps": -3.0}) == []
    assert sweep_follow_event_mean_bps_honesty_errors({"sweep_follow_event_mean_bps": 1.5}) == []
    assert sweep_follow_event_mean_bps_honesty_errors({}) == []


def test_follow_event_nan_skipped() -> None:
    assert (
        sweep_follow_event_mean_bps_honesty_errors({"sweep_follow_event_mean_bps": float("nan")})
        == []
    )


def test_follow_event_non_finite_and_non_numeric() -> None:
    assert "sweep_follow_event_mean_bps_non_finite" in (
        sweep_follow_event_mean_bps_honesty_errors({"sweep_follow_event_mean_bps": float("inf")})
    )
    assert "sweep_follow_event_mean_bps_non_numeric" in (
        sweep_follow_event_mean_bps_honesty_errors({"sweep_follow_event_mean_bps": "nope"})
    )
