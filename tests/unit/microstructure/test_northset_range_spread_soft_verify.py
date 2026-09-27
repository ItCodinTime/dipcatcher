"""Soft-verify corwin/abdi/roll/true_range/yz variance ≥0 when finite."""

from __future__ import annotations

from quant_fund.research.catalog import northset_range_spread_honesty_errors


def test_range_spread_ok() -> None:
    assert (
        northset_range_spread_honesty_errors(
            {
                "corwin_schultz_spread": 0.0,
                "abdi_ranaldo_spread": 0.02,
                "roll_spread": 0.01,
                "mean_true_range": 1.5,
                "yang_zhang_variance": 0.0001,
            }
        )
        == []
    )
    assert northset_range_spread_honesty_errors({}) == []


def test_range_spread_flags_negative() -> None:
    assert "mean_true_range_negative" in northset_range_spread_honesty_errors(
        {"mean_true_range": -0.1}
    )


def test_range_spread_flags_inf() -> None:
    assert "yang_zhang_variance_non_finite" in northset_range_spread_honesty_errors(
        {"yang_zhang_variance": float("inf")}
    )
