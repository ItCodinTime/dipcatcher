"""Soft-verify mean_notional_imbalance ∈ [-1,1]."""

from __future__ import annotations

from quant_fund.research.catalog import mean_notional_imbalance_honesty_errors


def test_mean_notional_ok() -> None:
    assert mean_notional_imbalance_honesty_errors({"mean_notional_imbalance": -0.2}) == []
    assert mean_notional_imbalance_honesty_errors({}) == []


def test_mean_notional_bad() -> None:
    assert mean_notional_imbalance_honesty_errors({"mean_notional_imbalance": 1.5}) == [
        "mean_notional_imbalance_out_of_unit_interval"
    ]
