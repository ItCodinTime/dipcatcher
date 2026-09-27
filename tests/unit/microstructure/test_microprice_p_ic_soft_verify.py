"""Soft-verify microprice_p_ic / microprice_t_ic finite when present."""

from __future__ import annotations

from quant_fund.research.catalog import microprice_p_ic_honesty_errors


def test_microprice_ic_ok() -> None:
    assert microprice_p_ic_honesty_errors({"microprice_p_ic": 0.1, "microprice_t_ic": -1.2}) == []
    assert microprice_p_ic_honesty_errors({}) == []


def test_microprice_ic_inf() -> None:
    assert "microprice_p_ic_non_finite_fail_closed" in microprice_p_ic_honesty_errors(
        {"microprice_p_ic": float("inf")}
    )
