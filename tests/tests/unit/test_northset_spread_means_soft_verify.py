"""Soft-verify northset mean_quoted/effective/spread_bps ≥0; quoted≈effective."""

from __future__ import annotations

from quant_fund.research.catalog import northset_spread_means_honesty_errors


def test_spread_means_ok() -> None:
    assert (
        northset_spread_means_honesty_errors(
            {
                "mean_quoted_spread": 0.01,
                "mean_effective_spread": 0.01,
                "mean_spread_bps": 4.0,
            }
        )
        == []
    )
    assert northset_spread_means_honesty_errors({}) == []
    assert northset_spread_means_honesty_errors({"mean_quoted_spread": float("nan")}) == []


def test_spread_means_flags_negative_and_mismatch() -> None:
    assert "mean_quoted_spread_negative" in northset_spread_means_honesty_errors(
        {"mean_quoted_spread": -0.01}
    )
    assert "mean_quoted_effective_spread_mismatch" in northset_spread_means_honesty_errors(
        {"mean_quoted_spread": 1.0, "mean_effective_spread": 2.0}
    )


def test_spread_bps_non_finite() -> None:
    assert "mean_spread_bps_non_finite" in northset_spread_means_honesty_errors(
        {"mean_spread_bps": float("inf")}
    )
