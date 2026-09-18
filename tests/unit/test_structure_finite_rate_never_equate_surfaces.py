"""Never equate candle finite_rate_* companions with northset structure aggregate.

Northset ``structure_finite_rate`` = nanmean(concentration/queue/side_notional/tob
finite rates). Candle ``structure_finite_rate`` = nanmean(finite_rate_* companions).
Same key name, different companion surfaces — soft-verify checks both families
independently without cross-requiring the other surface's keys.
"""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure import bench_candle_order_book
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import structure_finite_rate_honesty_errors

_CANDLE_COMPANIONS = (
    "finite_rate_microprice_minus_mid",
    "finite_rate_bid_size_concentration_top",
    "finite_rate_ask_size_concentration_top",
)

_NORTHSET_COMPANIONS = (
    "concentration_top_finite_rate",
    "queue_priority_finite_rate",
    "side_notional_finite_rate",
    "tob_size_share_finite_rate",
)


def _northset():
    bars = SyntheticMarketProvider(n_assets=4, n_days=40, seed=7).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    return bench_northset(bars, cfg)


def _candle():
    bars = SyntheticMarketProvider(n_assets=4, n_days=40, seed=7).get_bars()
    return bench_candle_order_book(bars, book=None, depth=5, seed=7, label="SYNTHETIC")


def test_northset_stamps_aggregate_not_candle_finite_rate_companions() -> None:
    ns = _northset()
    assert "structure_finite_rate" in ns
    for key in _CANDLE_COMPANIONS:
        assert key not in ns, f"northset must not stamp candle companion {key}"
    for key in _NORTHSET_COMPANIONS:
        assert key in ns


def test_candle_stamps_finite_rate_companions_not_northset_shape_rates() -> None:
    cb = _candle()
    assert "structure_finite_rate" in cb
    for key in _CANDLE_COMPANIONS:
        assert key in cb
    for key in _NORTHSET_COMPANIONS:
        assert key not in cb, f"candle must not stamp northset companion {key}"


def test_soft_verify_northset_aggregate_alone_ok_without_candle_companions() -> None:
    errs = structure_finite_rate_honesty_errors({"structure_finite_rate": 0.8})
    assert errs == []
    # Bad aggregate still fails without needing candle keys
    assert structure_finite_rate_honesty_errors({"structure_finite_rate": 1.2}) == [
        "structure_finite_rate_out_of_unit_interval"
    ]


def test_soft_verify_candle_companions_ok_without_northset_shape_rates() -> None:
    blob = {k: 1.0 for k in _CANDLE_COMPANIONS}
    blob["structure_finite_rate"] = 1.0
    assert structure_finite_rate_honesty_errors(blob) == []
