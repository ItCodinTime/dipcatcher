"""Soft-verify daily ohlc_identity_rate ∈ [0, 1]."""

from __future__ import annotations

from quant_fund.research.catalog import ohlc_identity_rate_honesty_errors


def test_ohlc_identity_ok() -> None:
    assert ohlc_identity_rate_honesty_errors({"ohlc_identity_rate": 0.0}) == []
    assert ohlc_identity_rate_honesty_errors({"ohlc_identity_rate": 1.0}) == []
    assert ohlc_identity_rate_honesty_errors({}) == []


def test_ohlc_identity_out_of_bounds() -> None:
    assert "ohlc_identity_rate_out_of_unit_interval" in ohlc_identity_rate_honesty_errors(
        {"ohlc_identity_rate": -0.01}
    )
    assert "ohlc_identity_rate_out_of_unit_interval" in ohlc_identity_rate_honesty_errors(
        {"ohlc_identity_rate": 1.01}
    )


def test_ohlc_identity_non_finite() -> None:
    assert ohlc_identity_rate_honesty_errors({"ohlc_identity_rate": float("nan")}) == []
    assert "ohlc_identity_rate_non_finite" in ohlc_identity_rate_honesty_errors(
        {"ohlc_identity_rate": float("inf")}
    )
