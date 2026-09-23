"""Soft-verify sweep_follow_signed_mean_ic finite when present."""

from __future__ import annotations

from quant_fund.research.catalog import sweep_follow_signed_mean_ic_honesty_errors


def test_sweep_follow_ic_ok() -> None:
    assert sweep_follow_signed_mean_ic_honesty_errors({"sweep_follow_signed_mean_ic": -0.05}) == []
    assert sweep_follow_signed_mean_ic_honesty_errors({}) == []


def test_sweep_follow_ic_inf() -> None:
    assert sweep_follow_signed_mean_ic_honesty_errors(
        {"sweep_follow_signed_mean_ic": float("inf")}
    ) == ["sweep_follow_signed_mean_ic_non_finite_fail_closed"]
