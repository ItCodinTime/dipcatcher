"""Soft-verify mean_microprice_minus_mid finite when present."""

from __future__ import annotations

from quant_fund.research.catalog import mean_microprice_minus_mid_honesty_errors


def test_mean_microprice_minus_mid_ok_and_skip() -> None:
    assert mean_microprice_minus_mid_honesty_errors({"mean_microprice_minus_mid": 0.01}) == []
    assert (
        mean_microprice_minus_mid_honesty_errors({"mean_microprice_minus_mid": float("nan")}) == []
    )
    assert mean_microprice_minus_mid_honesty_errors({}) == []


def test_mean_microprice_minus_mid_inf_fail() -> None:
    assert mean_microprice_minus_mid_honesty_errors(
        {"mean_microprice_minus_mid": float("inf")}
    ) == ["mean_microprice_minus_mid_non_finite_fail_closed"]
