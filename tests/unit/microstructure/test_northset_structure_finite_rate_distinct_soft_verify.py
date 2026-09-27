"""Soft-verify northset structure_finite_rate ≠ candle finite_rate_* pack."""

from __future__ import annotations

import inspect
import math

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    northset_structure_finite_rate_distinct_from_candle_honesty_errors,
)

_CANDLE = (
    "finite_rate_microprice_minus_mid",
    "finite_rate_bid_size_concentration_top",
    "finite_rate_ask_size_concentration_top",
)
_NORTH = (
    "concentration_top_finite_rate",
    "queue_priority_finite_rate",
    "side_notional_finite_rate",
    "tob_size_share_finite_rate",
)


def test_helper_wired_into_northset_receipt_dispatcher() -> None:
    assert northset_structure_finite_rate_distinct_from_candle_honesty_errors in (
        NORTHSET_RECEIPT_HONESTY_HELPERS
    )


def test_bench_source_uses_northset_companions_not_candle_finite_rate() -> None:
    src = inspect.getsource(bench_northset)
    assert '"structure_finite_rate"' in src
    for key in _NORTH:
        assert f'"{key}"' in src
    for key in _CANDLE:
        assert f'"{key}"' not in src


def test_synth_northset_receipt_structure_rate_distinct_and_identity() -> None:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    receipt = bench_northset(SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), cfg)
    for key in _CANDLE:
        assert key not in receipt
    for key in _NORTH:
        assert key in receipt
        assert math.isfinite(float(receipt[key]))
    agg = float(receipt["structure_finite_rate"])
    assert math.isfinite(agg) and 0.0 <= agg <= 1.0
    expected = sum(float(receipt[k]) for k in _NORTH) / 4.0
    assert math.isclose(agg, expected, rel_tol=1e-9, abs_tol=1e-12)
    assert northset_structure_finite_rate_distinct_from_candle_honesty_errors(receipt) == []


def test_honesty_rejects_mixed_candle_companions_and_bad_aggregate() -> None:
    base = {k: 1.0 for k in _NORTH}
    base["structure_finite_rate"] = 1.0
    assert northset_structure_finite_rate_distinct_from_candle_honesty_errors(base) == []
    mixed = dict(base)
    mixed["finite_rate_microprice_minus_mid"] = 0.5
    assert "northset_structure_finite_rate_mixed_with_candle_finite_rate_companions" in (
        northset_structure_finite_rate_distinct_from_candle_honesty_errors(mixed)
    )
    bad = dict(base)
    bad["structure_finite_rate"] = 0.5
    assert "northset_structure_finite_rate_not_nanmean_of_companion_rates" in (
        northset_structure_finite_rate_distinct_from_candle_honesty_errors(bad)
    )
