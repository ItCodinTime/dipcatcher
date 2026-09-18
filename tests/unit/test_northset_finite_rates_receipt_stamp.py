"""Northset/candle stamp gap_finite_rate, depth_shape_finite_rate, structure_finite_rate."""

from __future__ import annotations

import inspect
import math

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure import bench_candle_order_book
from quant_fund.northset import benches


def _synth_northset():
    bars = SyntheticMarketProvider(n_assets=4, n_days=40, seed=7).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    return benches.bench_northset(bars, cfg)


def _unit(val: float, key: str) -> None:
    assert not math.isinf(val), f"{key}={val}"
    if math.isfinite(val):
        assert 0.0 <= val <= 1.0, f"{key}={val}"


def test_bench_source_stamps_gap_depth_shape_and_structure_finite_rates() -> None:
    src = inspect.getsource(benches.bench_northset)
    assert '"gap_finite_rate"' in src
    assert '"depth_shape_finite_rate"' in src
    assert '"structure_finite_rate"' in src


def test_synth_northset_stamps_finite_rates_unit_interval() -> None:
    receipt = _synth_northset()
    for key in ("gap_finite_rate", "depth_shape_finite_rate", "structure_finite_rate"):
        assert key in receipt, key
        _unit(float(receipt[key]), key)


def test_synth_candle_stamps_structure_and_depth_shape_unit_interval() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=40, seed=7).get_bars()
    receipt = bench_candle_order_book(bars, book=None, depth=5, seed=7, label="SYNTHETIC")
    assert "depth_shape_finite_rate" in receipt
    assert "structure_finite_rate" in receipt
    _unit(float(receipt["depth_shape_finite_rate"]), "depth_shape_finite_rate")
    _unit(float(receipt["structure_finite_rate"]), "structure_finite_rate")
    # Derived from finite_rate_* companions
    for key in (
        "finite_rate_microprice_minus_mid",
        "finite_rate_bid_size_concentration_top",
        "finite_rate_ask_size_concentration_top",
    ):
        assert key in receipt


def test_synth_northset_structure_finite_rate_derived_from_shape_rates() -> None:
    """Northset aggregate = nanmean of concentration/queue/side_notional/tob finite rates."""
    import math

    receipt = _synth_northset()
    comps = [
        float(receipt["concentration_top_finite_rate"]),
        float(receipt["queue_priority_finite_rate"]),
        float(receipt["side_notional_finite_rate"]),
        float(receipt["tob_size_share_finite_rate"]),
    ]
    finite = [x for x in comps if math.isfinite(x)]
    assert finite
    expected = sum(finite) / len(finite)
    got = float(receipt["structure_finite_rate"])
    assert 0.0 <= got <= 1.0
    assert abs(got - expected) < 1e-12
