"""Soft-verify session_volume_conservation_rate ∈ [0,1]."""

from __future__ import annotations

from quant_fund.research.catalog import session_volume_conservation_rate_honesty_errors


def test_session_volume_conservation_ok() -> None:
    assert (
        session_volume_conservation_rate_honesty_errors({"session_volume_conservation_rate": 1.0})
        == []
    )
    assert (
        session_volume_conservation_rate_honesty_errors(
            {"session_volume_conservation_rate": float("nan")}
        )
        == []
    )


def test_session_volume_conservation_bad() -> None:
    assert session_volume_conservation_rate_honesty_errors(
        {"session_volume_conservation_rate": -0.1}
    ) == ["session_volume_conservation_rate_out_of_unit_interval"]
