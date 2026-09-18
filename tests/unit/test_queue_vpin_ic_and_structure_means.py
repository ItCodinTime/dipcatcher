"""queue/vpin IC packs + candle structure IC⇒mean honesty."""

from __future__ import annotations

import math
from pathlib import Path

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    candle_structure_ic_implies_mean_honesty_errors,
    northset_queue_imbalance_ic_honesty_errors,
    northset_vpin_ic_pack_honesty_errors,
)


def test_candle_structure_ic_implies_means() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    for key in (
        "mean_imbalance_top",
        "mean_queue_imbalance",
        "mean_tob_size_share",
        "mean_bid_size_concentration_top",
        "mean_ask_size_concentration_top",
    ):
        assert key in receipt
        val = float(receipt[key])
        assert math.isfinite(val)
    assert candle_structure_ic_implies_mean_honesty_errors(receipt) == []


def test_structure_ic_honesty_flags_missing_mean() -> None:
    assert "mean_imbalance_top_missing_while_ic_imbalance_top_scored" in (
        candle_structure_ic_implies_mean_honesty_errors(
            {"family": "candle_order_book", "ic_imbalance_top": 0.1}
        )
    )


def test_queue_and_vpin_ic_honesty_synth() -> None:
    assert "queue_imbalance_p_ic_out_of_unit_interval" in (
        northset_queue_imbalance_ic_honesty_errors({"queue_imbalance_p_ic": 1.5})
    )
    assert "vpin_p_ic_out_of_unit_interval" in (
        northset_vpin_ic_pack_honesty_errors({"vpin_p_ic": -0.01})
    )
    assert "vpin_n_dates_negative" in (northset_vpin_ic_pack_honesty_errors({"vpin_n_dates": -1}))
    assert northset_queue_imbalance_ic_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
    assert northset_vpin_ic_pack_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    receipt = bench_northset(SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), cfg)
    assert northset_queue_imbalance_ic_honesty_errors(receipt) == []
    assert northset_vpin_ic_pack_honesty_errors(receipt) == []


def test_verify_wires_structure_ic_mean_helper() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "candle_structure_ic_implies_mean_honesty_errors" in src
