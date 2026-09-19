"""mean_effective_spread ≥0; half-spread identity; FEATURE ofi finite."""

from __future__ import annotations

from pathlib import Path

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.research.catalog import (
    candle_feature_ofi_finite_honesty_errors,
    candle_spread_alias_honesty_errors,
)


def test_effective_spread_nonneg_and_half_identity() -> None:
    assert "mean_effective_spread_negative_or_non_finite" in (
        candle_spread_alias_honesty_errors(
            {"family": "candle_order_book", "mean_effective_spread": -0.01}
        )
    )
    bad = {
        "family": "candle_order_book",
        "mean_quoted_spread": 0.02,
        "mean_effective_spread": 0.02,
        "mean_half_spread": 0.02,  # should be 0.01
    }
    assert "mean_half_spread_not_half_of_mean_quoted_spread" in (
        candle_spread_alias_honesty_errors(bad)
    ) or any("half" in e for e in candle_spread_alias_honesty_errors(bad))


def test_ofi_finite_fail_closed() -> None:
    assert "mean_ofi_non_finite" in candle_feature_ofi_finite_honesty_errors(
        {"family": "candle_order_book", "mean_ofi": float("inf")}
    )


def test_synth_and_verify() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert candle_spread_alias_honesty_errors(receipt) == []
    assert candle_feature_ofi_finite_honesty_errors(receipt) == []
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "candle_spread_alias_honesty_errors" in src
    assert "candle_feature_ofi_finite_honesty_errors" in src
