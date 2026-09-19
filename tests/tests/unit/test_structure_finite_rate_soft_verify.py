"""Soft-verify structure_finite_rate + finite_rate_* ∈ [0,1]."""

from __future__ import annotations

from pathlib import Path

from quant_fund.research.catalog import structure_finite_rate_honesty_errors


def test_structure_finite_rate_companions_ok() -> None:
    assert (
        structure_finite_rate_honesty_errors(
            {
                "finite_rate_microprice_minus_mid": 1.0,
                "finite_rate_bid_size_concentration_top": 0.5,
                "finite_rate_ask_size_concentration_top": 0.0,
            }
        )
        == []
    )


def test_structure_finite_rate_aggregate_ok() -> None:
    assert structure_finite_rate_honesty_errors({"structure_finite_rate": 0.75}) == []
    assert structure_finite_rate_honesty_errors({}) == []


def test_structure_finite_rate_aggregate_bad() -> None:
    assert structure_finite_rate_honesty_errors({"structure_finite_rate": 1.5}) == [
        "structure_finite_rate_out_of_unit_interval"
    ]
    assert structure_finite_rate_honesty_errors({"structure_finite_rate": -0.1}) == [
        "structure_finite_rate_out_of_unit_interval"
    ]


def test_structure_finite_rate_companions_bad() -> None:
    errs = structure_finite_rate_honesty_errors({"finite_rate_microprice_minus_mid": 1.1})
    assert "finite_rate_microprice_minus_mid_out_of_unit_interval" in errs


def test_verify_wires_structure_finite_rate_on_northset_and_candle() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert 'structure_finite_rate_honesty_errors(families.get("northset"))' in src
    assert 'structure_finite_rate_honesty_errors(families.get("candle_order_book"))' in src
