"""Candle-book stamps finite_rate_* for structure_finite_rate_honesty."""

from __future__ import annotations

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.research.catalog import structure_finite_rate_honesty_errors

KEYS = (
    "finite_rate_microprice_minus_mid",
    "finite_rate_bid_size_concentration_top",
    "finite_rate_ask_size_concentration_top",
)


def test_synth_receipt_stamps_structure_finite_rates() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=40, seed=7).get_bars()
    rec = bench_candle_order_book(bars, book=None, depth=5, seed=7, label="SYNTHETIC")
    for key in KEYS:
        assert key in rec
        assert rec[key] == rec[key]  # finite on synth
        assert 0.0 <= float(rec[key]) <= 1.0
    assert structure_finite_rate_honesty_errors(rec) == []


def test_structure_finite_rate_equals_nanmean_of_finite_rate_companions() -> None:
    """Aggregate must be nanmean of the three finite_rate_* companions when all finite."""
    import math

    bars = SyntheticMarketProvider(n_assets=4, n_days=40, seed=7).get_bars()
    rec = bench_candle_order_book(bars, book=None, depth=5, seed=7, label="SYNTHETIC")
    comps = [
        float(rec["finite_rate_microprice_minus_mid"]),
        float(rec["finite_rate_bid_size_concentration_top"]),
        float(rec["finite_rate_ask_size_concentration_top"]),
    ]
    assert all(math.isfinite(x) for x in comps)
    expected = sum(comps) / len(comps)
    got = float(rec["structure_finite_rate"])
    assert 0.0 <= got <= 1.0
    assert abs(got - expected) < 1e-12
