"""Soft-verify mean_tob_size_share ∈ (0,1]."""

from __future__ import annotations

from quant_fund.research.catalog import mean_tob_size_share_honesty_errors


def test_mean_tob_ok() -> None:
    assert mean_tob_size_share_honesty_errors({"mean_tob_size_share": 0.4}) == []
    assert mean_tob_size_share_honesty_errors({}) == []


def test_mean_tob_bad() -> None:
    assert mean_tob_size_share_honesty_errors({"mean_tob_size_share": 0.0}) == [
        "mean_tob_size_share_out_of_open_unit_interval"
    ]
