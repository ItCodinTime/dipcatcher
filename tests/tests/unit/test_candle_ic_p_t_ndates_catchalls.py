"""Candle IC catch-alls: ic_*_p ∈[0,1], ic_*_t finite, ic_*_n_dates ≥0."""

from __future__ import annotations

from pathlib import Path

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.research.catalog import (
    candle_all_ic_n_dates_nonneg_honesty_errors,
    candle_all_ic_p_unit_honesty_errors,
    candle_all_ic_t_finite_honesty_errors,
)


def test_candle_ic_p_unit_ok_and_oob() -> None:
    ok = {"family": "candle_order_book", "ic_ofi_p": 0.12}
    assert candle_all_ic_p_unit_honesty_errors(ok) == []
    bad = {"family": "candle_order_book", "ic_ofi_p": 1.5}
    assert "ic_ofi_p_out_of_unit_interval" in candle_all_ic_p_unit_honesty_errors(bad)
    # pearson suffix must not be treated as _p
    assert (
        candle_all_ic_p_unit_honesty_errors({"family": "candle_order_book", "ic_ofi_pearson": 0.9})
        == []
    )


def test_candle_ic_t_and_n_dates() -> None:
    ok = {
        "family": "candle_order_book",
        "ic_ofi_t": 1.2,
        "ic_ofi_n_dates": 10,
    }
    assert candle_all_ic_t_finite_honesty_errors(ok) == []
    assert candle_all_ic_n_dates_nonneg_honesty_errors(ok) == []
    assert "ic_ofi_t_non_finite_fail_closed" in candle_all_ic_t_finite_honesty_errors(
        {"family": "candle_order_book", "ic_ofi_t": float("inf")}
    )
    assert "ic_ofi_n_dates_negative_or_non_finite" in (
        candle_all_ic_n_dates_nonneg_honesty_errors(
            {"family": "candle_order_book", "ic_ofi_n_dates": -1}
        )
    )


def test_synth_candle_ic_catchalls_clean() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert candle_all_ic_p_unit_honesty_errors(receipt) == []
    assert candle_all_ic_t_finite_honesty_errors(receipt) == []
    assert candle_all_ic_n_dates_nonneg_honesty_errors(receipt) == []


def test_verify_wires_candle_ic_catchalls() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "candle_all_ic_p_unit_honesty_errors" in src
    assert "candle_all_ic_t_finite_honesty_errors" in src
    assert "candle_all_ic_n_dates_nonneg_honesty_errors" in src
