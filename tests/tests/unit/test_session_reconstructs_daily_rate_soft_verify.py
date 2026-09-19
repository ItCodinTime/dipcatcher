"""Soft-verify session_reconstructs_daily_rate ∈ [0,1]."""

from __future__ import annotations

from quant_fund.research.catalog import session_reconstructs_daily_rate_honesty_errors


def test_session_reconstructs_daily_ok() -> None:
    assert (
        session_reconstructs_daily_rate_honesty_errors({"session_reconstructs_daily_rate": 0.95})
        == []
    )
    assert (
        session_reconstructs_daily_rate_honesty_errors(
            {"session_reconstructs_daily_rate": float("nan")}
        )
        == []
    )


def test_session_reconstructs_daily_bad() -> None:
    assert session_reconstructs_daily_rate_honesty_errors(
        {"session_reconstructs_daily_rate": 1.5}
    ) == ["session_reconstructs_daily_rate_out_of_unit_interval"]
