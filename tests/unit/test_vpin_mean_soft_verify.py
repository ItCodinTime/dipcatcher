"""Soft-verify vpin_mean ∈ [0, 1] when finite."""

from __future__ import annotations

from quant_fund.research.catalog import vpin_mean_honesty_errors


def test_vpin_mean_unit_interval() -> None:
    assert vpin_mean_honesty_errors({"vpin_mean": 0.0}) == []
    assert vpin_mean_honesty_errors({"vpin_mean": 1.0}) == []
    assert vpin_mean_honesty_errors({"vpin_mean": 0.33}) == []
    assert vpin_mean_honesty_errors({"vpin_mean": float("nan")}) == []
    assert vpin_mean_honesty_errors({}) == []


def test_vpin_mean_flags_oob() -> None:
    assert vpin_mean_honesty_errors({"vpin_mean": 1.1}) == ["vpin_mean_out_of_unit_interval"]
    assert vpin_mean_honesty_errors({"vpin_mean": -0.01}) == ["vpin_mean_out_of_unit_interval"]


def test_vpin_mean_rejects_inf() -> None:
    assert vpin_mean_honesty_errors({"vpin_mean": float("inf")}) == [
        "vpin_mean_out_of_unit_interval"
    ]
