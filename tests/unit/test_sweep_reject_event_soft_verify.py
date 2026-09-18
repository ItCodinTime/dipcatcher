"""Soft-verify sweep_reject_event_mean_bps finite when present."""

from __future__ import annotations

from quant_fund.research.catalog import sweep_reject_event_mean_bps_honesty_errors


def test_sweep_reject_ok() -> None:
    assert sweep_reject_event_mean_bps_honesty_errors({"sweep_reject_event_mean_bps": -12.5}) == []
    assert sweep_reject_event_mean_bps_honesty_errors({}) == []


def test_sweep_reject_inf() -> None:
    assert sweep_reject_event_mean_bps_honesty_errors(
        {"sweep_reject_event_mean_bps": float("inf")}
    ) == ["sweep_reject_event_mean_bps_non_finite_fail_closed"]
