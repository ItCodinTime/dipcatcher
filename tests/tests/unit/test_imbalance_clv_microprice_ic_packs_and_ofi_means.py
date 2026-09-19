"""imbalance_top/CLV/microprice IC packs + candle ofi/QP/slope IC⇒mean."""

from __future__ import annotations

import math
from pathlib import Path

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    candle_ofi_qp_slope_ic_implies_mean_honesty_errors,
    northset_clv_ic_pack_honesty_errors,
    northset_imbalance_top_ic_pack_honesty_errors,
    northset_microprice_bps_ic_pack_honesty_errors,
)


def test_candle_ofi_qp_slope_means_and_honesty() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    for key in (
        "mean_ofi",
        "mean_queue_priority_proxy",
        "mean_ask_queue_priority_proxy",
        "mean_bid_log_size_slope",
        "mean_ask_log_size_slope",
    ):
        assert key in receipt
        assert math.isfinite(float(receipt[key]))
    qp = float(receipt["mean_queue_priority_proxy"])
    assert 0.0 <= qp <= 1.0
    assert candle_ofi_qp_slope_ic_implies_mean_honesty_errors(receipt) == []


def test_ic_packs_bounds_and_synth() -> None:
    assert "imbalance_top_p_ic_out_of_unit_interval" in (
        northset_imbalance_top_ic_pack_honesty_errors({"imbalance_top_p_ic": 1.2})
    )
    assert "clv_p_ic_out_of_unit_interval" in (
        northset_clv_ic_pack_honesty_errors({"clv_p_ic": -0.1})
    )
    assert "microprice_minus_mid_bps_p_ic_out_of_unit_interval" in (
        northset_microprice_bps_ic_pack_honesty_errors({"microprice_minus_mid_bps_p_ic": 2.0})
    )
    for fn in (
        northset_imbalance_top_ic_pack_honesty_errors,
        northset_clv_ic_pack_honesty_errors,
        northset_microprice_bps_ic_pack_honesty_errors,
    ):
        assert fn in NORTHSET_RECEIPT_HONESTY_HELPERS
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    receipt = bench_northset(SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), cfg)
    assert northset_imbalance_top_ic_pack_honesty_errors(receipt) == []
    assert northset_clv_ic_pack_honesty_errors(receipt) == []
    assert northset_microprice_bps_ic_pack_honesty_errors(receipt) == []


def test_verify_wires_ofi_qp_slope_helper() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "candle_ofi_qp_slope_ic_implies_mean_honesty_errors" in src
