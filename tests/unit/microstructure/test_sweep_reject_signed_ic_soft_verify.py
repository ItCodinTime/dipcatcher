"""Soft-verify sweep_reject_signed_mean_ic finite when present."""

from __future__ import annotations

from quant_fund.research.catalog import sweep_reject_signed_mean_ic_honesty_errors


def test_sweep_reject_ic_ok() -> None:
    assert sweep_reject_signed_mean_ic_honesty_errors({"sweep_reject_signed_mean_ic": 0.12}) == []
    assert sweep_reject_signed_mean_ic_honesty_errors({}) == []


def test_sweep_reject_ic_inf() -> None:
    assert sweep_reject_signed_mean_ic_honesty_errors(
        {"sweep_reject_signed_mean_ic": float("-inf")}
    ) == ["sweep_reject_signed_mean_ic_non_finite_fail_closed"]
