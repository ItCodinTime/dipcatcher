"""Soft-verify session_mean_jump_ratio ∈ [0, 1]."""

from __future__ import annotations

from quant_fund.research.catalog import session_mean_jump_ratio_honesty_errors


def test_jump_ratio_ok() -> None:
    assert session_mean_jump_ratio_honesty_errors({"session_mean_jump_ratio": 0.0}) == []
    assert session_mean_jump_ratio_honesty_errors({"session_mean_jump_ratio": 1.0}) == []
    assert session_mean_jump_ratio_honesty_errors({}) == []


def test_jump_ratio_out_of_bounds() -> None:
    assert "session_mean_jump_ratio_out_of_unit_interval" in session_mean_jump_ratio_honesty_errors(
        {"session_mean_jump_ratio": -0.01}
    )
    assert "session_mean_jump_ratio_out_of_unit_interval" in session_mean_jump_ratio_honesty_errors(
        {"session_mean_jump_ratio": 1.01}
    )


def test_jump_ratio_non_finite() -> None:
    assert session_mean_jump_ratio_honesty_errors({"session_mean_jump_ratio": float("nan")}) == []
    assert "session_mean_jump_ratio_non_finite" in session_mean_jump_ratio_honesty_errors(
        {"session_mean_jump_ratio": float("inf")}
    )
