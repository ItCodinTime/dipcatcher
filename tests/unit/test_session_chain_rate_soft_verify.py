"""Soft-verify session_chain_rate ∈ [0,1]."""

from __future__ import annotations

from quant_fund.research.catalog import session_chain_rate_honesty_errors


def test_session_chain_rate_ok() -> None:
    assert session_chain_rate_honesty_errors({"session_chain_rate": 0.0}) == []
    assert session_chain_rate_honesty_errors({"session_chain_rate": 1.0}) == []
    assert session_chain_rate_honesty_errors({"session_chain_rate": float("nan")}) == []


def test_session_chain_rate_bad() -> None:
    assert session_chain_rate_honesty_errors({"session_chain_rate": 1.1}) == [
        "session_chain_rate_out_of_unit_interval"
    ]
