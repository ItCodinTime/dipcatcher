"""Soft-verify mean_close_mid_abs_rel ≥ 0."""

from __future__ import annotations

from quant_fund.research.catalog import mean_close_mid_abs_rel_honesty_errors


def test_close_mid_abs_rel_ok() -> None:
    assert mean_close_mid_abs_rel_honesty_errors({"mean_close_mid_abs_rel": 0.0}) == []
    assert mean_close_mid_abs_rel_honesty_errors({}) == []


def test_close_mid_abs_rel_negative() -> None:
    assert "mean_close_mid_abs_rel_negative" in mean_close_mid_abs_rel_honesty_errors(
        {"mean_close_mid_abs_rel": -1e-6}
    )


def test_close_mid_abs_rel_non_finite() -> None:
    assert mean_close_mid_abs_rel_honesty_errors({"mean_close_mid_abs_rel": float("nan")}) == []
    assert "mean_close_mid_abs_rel_non_finite" in mean_close_mid_abs_rel_honesty_errors(
        {"mean_close_mid_abs_rel": float("inf")}
    )
