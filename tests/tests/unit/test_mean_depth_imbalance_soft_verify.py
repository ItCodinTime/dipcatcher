"""Soft-verify mean_depth_imbalance ∈ [-1,1]."""

from __future__ import annotations

from quant_fund.research.catalog import mean_depth_imbalance_honesty_errors


def test_mean_depth_ok() -> None:
    assert mean_depth_imbalance_honesty_errors({"mean_depth_imbalance": 0.3}) == []
    assert mean_depth_imbalance_honesty_errors({}) == []


def test_mean_depth_bad() -> None:
    assert mean_depth_imbalance_honesty_errors({"mean_depth_imbalance": -1.1}) == [
        "mean_depth_imbalance_out_of_unit_interval"
    ]
