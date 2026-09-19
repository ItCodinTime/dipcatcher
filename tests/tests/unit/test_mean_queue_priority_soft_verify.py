"""Soft-verify mean_queue_priority_proxy ∈ [0,1]."""

from __future__ import annotations

from quant_fund.research.catalog import mean_queue_priority_honesty_errors


def test_mean_queue_priority_ok() -> None:
    assert mean_queue_priority_honesty_errors({"mean_queue_priority_proxy": 0.5}) == []
    assert mean_queue_priority_honesty_errors({}) == []


def test_mean_queue_priority_bad() -> None:
    assert mean_queue_priority_honesty_errors({"mean_queue_priority_proxy": 1.2}) == [
        "mean_queue_priority_proxy_out_of_unit_interval"
    ]


def test_ask_queue_priority_bounds() -> None:
    from quant_fund.research.catalog import mean_queue_priority_honesty_errors

    assert mean_queue_priority_honesty_errors({"mean_ask_queue_priority_proxy": 0.25}) == []
    assert "mean_ask_queue_priority_proxy_out_of_unit_interval" in (
        mean_queue_priority_honesty_errors({"mean_ask_queue_priority_proxy": 1.2})
    )
